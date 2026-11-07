"""Yggdrasil inference on a GPU in one fused Triton kernel per token step, float32, the interface of inference.py.

    model = Yggdrasil.load("results/<run>.npz", backend="triton")   # yggdrasil.inference

One program per stream runs every layer in order: the hard routing of each routed layer on
f = [router_embed(x); rms(h_below)], the expert reads, h <- A_r h + T_r[x] + U'_r V' rms(h_below) and the readout P_r h.
The unembedding is one GEMM in torch. The state is a (B, layers, n) tensor on the device.
"""
import os

os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")  # fp32 dots without TF32
import numpy as np
import torch
import triton
import triton.language as tl

from .inference import LN2, router_depth  # noqa: E402


# LOOK (router levels read per round of loads) and num_warps are tuned per batch size at the first step.
@triton.jit
def _ygg_step(x_ptr, h_in, h_out, z_out, expert_out,
              tk_table, tk_A, tk_readout,
              rt_embed, rt_weight, rt_bias, rt_table, rt_A, rt_lin_proj, rt_lin_down, rt_readout, V, P,
              N: tl.constexpr, G: tl.constexpr, BS: tl.constexpr, RO: tl.constexpr, ROC: tl.constexpr,
              RL: tl.constexpr, RLC: tl.constexpr, F: tl.constexpr, FP: tl.constexpr, FIN: tl.constexpr,
              ROUTE_ON_STATE: tl.constexpr, D: tl.constexpr, K: tl.constexpr,
              TOKEN_A: tl.constexpr, ROUTED_A: tl.constexpr, HAS_BELOW: tl.constexpr, LOOK: tl.constexpr):
    s = tl.program_id(0).to(tl.int64)
    x = tl.load(x_ptr + s).to(tl.int64)
    gi = tl.arange(0, G)
    bi = tl.arange(0, BS)
    idx = gi[:, None] * BS + bi[None, :]                                          # (G, BS): state index
    aidx = gi[:, None, None] * (BS * BS) + bi[None, :, None] * BS + bi[None, None, :]   # (G, BS, BS)
    ro = tl.arange(0, ROC)

    # token layer: expert = token
    expert = x
    h = tl.load(tk_table + (expert * V + x) * N + idx)
    if TOKEN_A:
        hp = tl.load(h_in + (s * P) * N + idx)
        a = tl.load(tk_A + expert * (N * BS) + aidx)
        h += tl.sum(a * hp[:, None, :], 2)
    tl.store(h_out + (s * P) * N + idx, h)
    tl.store(expert_out + s * P, expert.to(tl.int32))
    for c in tl.static_range(0, RO, ROC):
        po = tl.load(tk_readout + (expert * RO + c + ro)[:, None, None] * N + idx[None, :, :])
        tl.store(z_out + (s * P) * RO + c + ro, tl.sum(tl.sum(po * h[None, :, :], 2), 1))
    below = h * tl.rsqrt(tl.sum(tl.sum(h * h, 1), 0) / N + 1e-6)  # config.RMS_EPS

    fi = tl.arange(0, FP)
    fmask = fi < F
    EXPERTS: tl.constexpr = 1 << D
    for k in range(K):
        part = k + 1
        kk = s * 0 + k
        # hard routing
        tf = tl.load(rt_embed + (kk * V + x) * F + fi, mask=fmask, other=0.0)
        w = rt_weight + kk * ((EXPERTS - 1) * FIN)
        node = x * 0
        J: tl.constexpr = 1 << LOOK
        j = tl.arange(0, J)
        dj = j * 0                                                               # depth of j in a LOOK-level subtree
        for l in tl.static_range(1, LOOK):
            dj += (j + 1 >= (1 << l)).to(tl.int32)
        for c0 in tl.static_range(0, D, LOOK):
            # the LOOK-level subtree under node in one round of loads: heap index j -> node * 2^depth(j) + j
            gnode = node * (1 << dj).to(tl.int64) + j
            valid = (j < J - 1) & (c0 + dj < D)
            sc = tl.sum(tl.load(w + gnode[:, None] * FIN + fi[None, :], mask=valid[:, None] & fmask[None, :],
                                other=0.0) * tf[None, :], 1)
            if ROUTE_ON_STATE:
                wl = tl.load(w + gnode[:, None, None] * FIN + F + idx[None, :, :], mask=valid[:, None, None],
                             other=0.0)
                sc += tl.sum(tl.sum(wl * below[None, :, :], 2), 1)
            sc += tl.load(rt_bias + kk * (EXPERTS - 1) + gnode, mask=valid, other=0.0)
            r = x * 0
            mult = x * 0 + 1
            for l in tl.static_range(LOOK):
                if c0 + l < D:
                    sr = tl.sum(tl.where(j == r, sc, 0.0), 0)
                    r = 2 * r + 1 + (sr > 0).to(tl.int64)
                    mult = mult * 2
            node = node * mult + r
        expert = node - (EXPERTS - 1)
        tl.store(expert_out + s * P + part, expert.to(tl.int32))
        gl = kk * EXPERTS + expert                                               # row in the stacked experts
        drive = tl.load(rt_table + (gl * V + x) * N + idx)
        if HAS_BELOW:
            for c in tl.static_range(0, RL, RLC):
                r = c + tl.arange(0, RLC)
                vl = tl.load(rt_lin_down + kk * (RL * N) + r[:, None, None] * N + idx[None, :, :])
                lv = tl.sum(tl.sum(vl * below[None, :, :], 2), 1)                  # (RLC,)
                ul = tl.load(rt_lin_proj + gl * (N * RL) + idx[:, :, None] * RL + r[None, None, :])
                drive += tl.sum(ul * lv[None, None, :], 2)
        h = drive
        if ROUTED_A:
            hp = tl.load(h_in + (s * P + part) * N + idx)
            a = tl.load(rt_A + gl * (N * BS) + aidx)
            h += tl.sum(a * hp[:, None, :], 2)
        tl.store(h_out + (s * P + part) * N + idx, h)
        for c in tl.static_range(0, RO, ROC):
            row = (kk * EXPERTS + expert) * RO + c + ro
            po = tl.load(rt_readout + row[:, None, None] * N + idx[None, :, :])
            tl.store(z_out + (s * P + part) * RO + c + ro, tl.sum(tl.sum(po * h[None, :, :], 2), 1))
        below = h * tl.rsqrt(tl.sum(tl.sum(h * h, 1), 0) / N + 1e-6)


def _transitions(p):
    """A_r in float64, then float32."""
    sk = torch.as_tensor(p["rotation"], dtype=torch.float64)
    sk = 0.5 * (sk - sk.transpose(-1, -2))
    d = torch.as_tensor(p["decay"], dtype=torch.float64).clamp(-1.0, 1.0).reshape(sk.shape[:-1])
    return (d[..., None] * torch.linalg.matrix_exp(sk)).float().contiguous()


def _table(p):
    """T_r[x] = U_r v(x) + c_r, (experts, V, n)."""
    u, v, c = (torch.as_tensor(p[k], dtype=torch.float64) for k in ("input_proj", "token_embed", "bias"))
    return (torch.einsum("lkr,vr->lvk", u, v) + c[:, None, :]).float().contiguous()


class TritonYggdrasil:
    def __init__(self, layers, output_bias, device="cuda", cuda_graph=False):
        """layers, output_bias as from inference.read(). cuda_graph: replay one captured step per batch size."""
        tk, rt = layers[0], layers[1:]
        if not rt or len({"rotation" in p for p in rt}) > 1:
            raise NotImplementedError("the fused step needs at least one routed layer, all of one kind")
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device).contiguous()
        st = lambda k: torch.stack([t(p[k]) for p in rt]).contiguous()
        none = torch.zeros(1, dtype=torch.float32, device=device)
        r0 = rt[0]
        self.device, self.cuda_graph, self._graphs, self._config = device, cuda_graph, {}, {}
        self.layers, self.n = len(layers), tk["bias"].shape[1]
        # token ids index the tables (vocab), the logits are over the classes
        self.vocab, self.classes = tk["token_embed"].shape[0], len(output_bias)
        self.depth = router_depth(r0)
        self.token_A, self.routed_A, self.has_below = "rotation" in tk, "rotation" in r0, "layer_input_down" in r0
        self.bs = (tk if self.token_A else r0)["rotation"].shape[-1] if self.token_A or self.routed_A else self.n
        self.ro, self.rl = tk["readout"].shape[1], r0["layer_input_down"].shape[0] if self.has_below else 1
        self.f, self.fin = r0["router_embed"].shape[1], r0["router_weight"].shape[1]
        self.tk_table, self.tk_readout = _table(tk).to(device), t(tk["readout"])
        self.tk_A = _transitions(tk).to(device) if self.token_A else none
        self.rt_embed, self.rt_weight, self.rt_bias = st("router_embed"), st("router_weight"), st("router_bias")
        self.rt_table = torch.stack([_table(p) for p in rt]).to(device).contiguous()
        self.rt_A = torch.stack([_transitions(p) for p in rt]).to(device).contiguous() if self.routed_A else none
        self.rt_lin_proj, self.rt_lin_down = (st("layer_input_proj"), st("layer_input_down")) if self.has_below \
            else (none, none)
        self.rt_readout = st("readout")
        self.unembed = torch.cat([t(p["unembed"]).T for p in layers]).contiguous()  # logits = bias + [z_0, z_1, ...] Q
        self.output_bias = t(output_bias)

    def init_state(self, batch_size):
        """Zero state of batch_size streams, (B, layers, n)."""
        return torch.zeros(batch_size, self.layers, self.n, device=self.device)

    def _launch(self, x, h, h_out, z, experts, warps, look):
        _ygg_step[(len(x),)](x, h, h_out, z, experts, self.tk_table, self.tk_A, self.tk_readout,
                             self.rt_embed, self.rt_weight, self.rt_bias, self.rt_table, self.rt_A, self.rt_lin_proj,
                             self.rt_lin_down, self.rt_readout, self.vocab, self.layers,
                             N=self.n, G=self.n // self.bs, BS=self.bs, RO=self.ro, ROC=min(self.ro, 8), RL=self.rl,
                             RLC=min(self.rl, 16), F=self.f, FP=triton.next_power_of_2(self.f), FIN=self.fin,
                             ROUTE_ON_STATE=self.fin != self.f, D=self.depth, K=self.layers - 1,
                             TOKEN_A=self.token_A, ROUTED_A=self.routed_A, HAS_BELOW=self.has_below, LOOK=look,
                             num_warps=warps)

    def _tune(self, B, steps=20):
        """(num_warps, router levels per round of loads) of the fastest kernel at batch size B, as triton.autotune."""
        x, h = torch.randint(0, self.vocab, (B,), device=self.device), self.init_state(B)
        bufs = torch.empty_like(h), torch.empty(B, self.layers, self.ro, device=self.device), \
            torch.empty(B, self.layers, dtype=torch.int32, device=self.device)
        best = None
        for warps in (1, 2, 4, 8):
            for look in (1, 2, 3):
                for _ in range(3):
                    self._launch(x, h, *bufs, warps, look)
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                for _ in range(steps):
                    self._launch(x, h, *bufs, warps, look)
                e1.record()
                torch.cuda.synchronize()
                if best is None or e0.elapsed_time(e1) < best[0]:
                    best = e0.elapsed_time(e1), warps, look
        return best[1:]

    def _step(self, x, h):
        B = len(x)
        if B not in self._config:
            self._config[B] = self._tune(B)
        h_out, z = torch.empty_like(h), torch.empty(B, self.layers, self.ro, device=self.device)
        self._launch(x, h, h_out, z, torch.empty(B, self.layers, dtype=torch.int32, device=self.device),
                     *self._config[B])
        return torch.addmm(self.output_bias, z.view(B, -1), self.unembed), h_out

    def step(self, tokens, state):
        """One token per stream: tokens (B,) -> logits (B, classes) of the next token, state."""
        x, state = torch.as_tensor(tokens, device=self.device).contiguous(), state.contiguous()  # the kernel reads flat
        if not self.cuda_graph:
            return self._step(x, state)
        if len(x) not in self._graphs:
            sx, sh = torch.zeros(len(x), dtype=torch.int64, device=self.device), self.init_state(len(x))
            self._step(sx, sh)  # tunes the kernel for this batch size before the capture
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = self._step(sx, sh)
            self._graphs[len(x)] = g, sx, sh, out
        g, sx, sh, (logits, h) = self._graphs[len(x)]
        sx.copy_(x)
        sh.copy_(state)
        g.replay()
        return logits.clone(), h.clone()

    @torch.no_grad()
    def score(self, tokens, state=None, batch=1024):
        """Bits of tokens[:, 1:] given the tokens before them, per row of tokens (B, L), as float64 (B,)."""
        tokens = np.atleast_2d(np.asarray(tokens, np.int64))
        out = []
        for i in range(0, len(tokens), batch):
            x = torch.as_tensor(tokens[i:i + batch], device=self.device)
            st = self.init_state(len(x)) if state is None else state[i:i + batch]
            nll = torch.zeros(len(x), dtype=torch.float64, device=self.device)
            for t in range(x.shape[1] - 1):
                logits, st = self.step(x[:, t], st)
                nll -= torch.log_softmax(logits, -1).gather(1, x[:, t + 1, None])[:, 0].double()
            out.append((nll / LN2).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def generate(self, prompt, n, temperature=1.0, top_k=0, seed=0):
        """Prompt (L,) or (B, L), L >= 1, followed by n sampled tokens. temperature 0: greedy."""
        prompt = np.asarray(prompt, np.int64)
        x = torch.as_tensor(np.atleast_2d(prompt), device=self.device)
        state, gen, out = self.init_state(len(x)), torch.Generator(self.device).manual_seed(seed), [x]
        for t in range(x.shape[1]):
            logits, state = self.step(x[:, t], state)
        for i in range(n):
            if temperature == 0:
                tok = logits.argmax(-1)
            else:
                if top_k:
                    logits = logits.masked_fill(logits < logits.topk(top_k, -1).values[:, -1:], -float("inf"))
                tok = torch.multinomial(torch.softmax(logits / temperature, -1), 1, generator=gen)[:, 0]
            out.append(tok[:, None])
            if i + 1 < n:
                logits, state = self.step(tok, state)
        out = torch.cat(out, 1).cpu().numpy()
        return out[0] if prompt.ndim == 1 else out
