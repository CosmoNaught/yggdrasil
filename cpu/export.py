"""Export a checkpoint (.npz) to model.bin for ygg.cpp, and a bit-exact NumPy model of the kernel (int_forward).

    python cpu/export.py --npz results/<run>.npz --out model.bin [--storage factored]
    python cpu/export.py --random --d_state 128 --block_size 16 --router_depth 8 --vocab 256 --out model.bin

The integer scheme is in cpu/README.md, the file layout in FORMAT below.
"""
import argparse
import dataclasses
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tasks.text import DATA, streams, windows  # noqa: E402
from yggdrasil import inference  # noqa: E402
from yggdrasil.config import RMS_EPS, upgrade_config  # noqa: E402

f32 = np.float32
SHIFT = 14                         # fixed-point shift of the transitions: A16 = rint(2^SHIFT A_r)
INT8_MAX = 127
SCALE_FLOOR = 1e-12                # smallest quantisation range
STATE_GROUP = 16                   # coordinates per state scale with group scales
GROUP_SCALE_FLOOR = 1e-3           # group scale floor, relative to the largest group
EXPERT_CHUNK = 256                 # experts per quantisation pass over the tables (memory)
CALIB_SKIP_TOKENS = 1_000_000      # calibration windows start here in the training split
HMARGIN_GRID = (1.0, 0.85, 0.7, 0.6)  # state range margins tried by calibration bits

# ------------------------------------------------------------------------------------------------- file format
# "YGG1", int32 HEADER fields, int32 array count, the array table [(name[32], offset int64, bytes int64)], then the
# arrays, each 64-byte aligned. Per layer l the arrays are named l<l>_<tag>; l<l>_meta holds the record bytes and the
# byte offsets of the RECORD parts inside each expert's record (-1: absent), l<l>_scal the SCALARS. ygg.cpp reads
# the same names; its Header struct mirrors HEADER.
MAGIC = b"YGG1"
OUTPUT_BIAS = "head_b"             # array name of the output bias (V padded to CLASS_PAD), fixed by format YGG1
HEADER = ("n_layers", "d_state", "block_size", "vocab", "router_depth", "readout_rank", "layer_input_rank",
          "reserved", "router_token_dim", "input_rank", "shift")  # reserved: always 0
RECORD = ("transition", "layer_input_proj", "readout", "table", "input_proj", "bias", "input_scale")
RECORD_TAGS = dict(zip(RECORD, "AUPTFCS"))
FACTORED_PARTS = 3                 # input_proj, bias, input_scale: in l<l>_meta only with factored storage
SCALARS = ("readout_scale", "unembed_scale", "rms_eps", "layer_input_scale")  # 0 where an array replaces them
LIMITS = dict(d_state=(64, 128, 256), block_size=(8, 16), readout_rank=(8, 16, 32), max_layer_input_rank=64,
              max_input_rank=64, max_vocab=256, max_layers=8, max_router_depth=16)
RANK_MULTIPLE = 4                  # input ranks are read in groups of 4 bytes
CLASS_PAD = 32                     # classes are padded to a multiple of 32 (two 16-lane chunks)
ALIGN = 64


def round_up(x, m):
    return (x + m - 1) // m * m


# ------------------------------------------------------------------------------------------------- float model

def bpc(logits, y):
    """Bits per token of targets y under logits."""
    lg = logits.astype(np.float64)
    mx = lg.max(-1, keepdims=True)
    lse = np.log(np.exp(lg - mx).sum(-1)) + mx[..., 0]
    return float((lse - np.take_along_axis(lg, y[..., None], -1)[..., 0]).mean() / np.log(2))


@dataclasses.dataclass
class FloatLayer:
    """One layer of the float64 inference model; E experts (vocab for the token layer), V classes."""
    token_layer: bool
    transition: np.ndarray            # (E, n/b, b, b)
    table: np.ndarray                 # (E, V, n) T_r[x] = U_r v(x) + c_r
    input_proj: np.ndarray            # (E, n, input_rank)
    token_embed: np.ndarray           # (V, input_rank)
    bias: np.ndarray                  # (E, n)
    readout: np.ndarray               # (E, R, n)
    unembed: np.ndarray               # (V, R)
    router_embed: np.ndarray = None   # (V, F)
    router_weight: np.ndarray = None  # (2^D - 1, F + n)
    router_bias: np.ndarray = None    # (2^D - 1,)
    layer_input_down: np.ndarray = None  # (R2, n)
    layer_input_proj: np.ndarray = None  # (E, n, R2)
    depth: int = 0

    @property
    def n(self):
        return self.bias.shape[1]

    @property
    def block(self):
        return self.transition.shape[-1]

    @property
    def router_token_dim(self):
        return 0 if self.token_layer else self.router_embed.shape[1]


def hard_layers(path):
    """Checkpoint -> the inference model in float64, its FloatLayers, and the output bias. Float64 makes the
    exported file the same on every host (float32 results of expm and matmuls differ between CPUs)."""
    import jax
    raw, output_bias = inference.read(path)
    raw = [{k: v.astype(np.float64) for k, v in p.items()} for p in raw]
    with jax.enable_x64(True):
        model = inference.Yggdrasil(raw, output_bias.astype(np.float64))
    layers = []
    for i, (p, q) in enumerate(zip(raw, model.params["layers"])):
        if "transition" not in q:
            raise NotImplementedError("experts without transitions (expert_type output) have no int8 export")
        if i and q["router_weight"].shape[1] == q["router_embed"].shape[1]:
            raise NotImplementedError("routing on the token only (router_input token) has no int8 export")
        q = {k: np.asarray(v) for k, v in q.items()}
        layers.append(FloatLayer(token_layer=i == 0, depth=model.depth if i else 0,
                                 **{k: p[k] for k in ("input_proj", "token_embed", "bias")}, **q))
    return model, layers, output_bias


@dataclasses.dataclass
class LayerStats:
    """Calibration maxima of one layer."""
    state_max: float                  # |h|
    layer_input_max: float            # |V' rms(h_below)|
    readout_max: float                # |P_r h|
    state_group_max: np.ndarray       # |h| per STATE_GROUP coordinates
    readout_row_max: np.ndarray       # |P_r h| per row


def calibrate(model, x):
    """LayerStats of every layer over windows x (S, T)."""
    import jax
    import jax.numpy as jnp

    @jax.jit
    def run(params, x):
        def f(state, xt):
            below, new, seen = None, [], []
            for i, p in enumerate(params["layers"]):
                r = xt if i == 0 else inference.route(p, xt, below)
                v = jnp.abs(below @ p["layer_input_down"].T).max() if "layer_input_down" in p else \
                    jnp.zeros((), jnp.float64)
                h, z = inference.update(p, r, xt, state[i], below)
                hg, zr = jnp.abs(h).reshape(len(h), -1, STATE_GROUP).max((0, 2)), jnp.abs(z).max(0)
                seen.append((hg.max(), v, zr.max(), hg, zr))
                new.append(h)
                below = inference.rms(h)
            return jnp.stack(new), seen
        state = jnp.zeros((model.layers, len(x), model.n), jnp.float64)
        with jax.default_matmul_precision("highest"):
            return jax.tree_util.tree_map(lambda a: a.max(0), jax.lax.scan(f, state, x.T)[1])
    with jax.enable_x64(True):
        st = jax.device_get(run(model.params, jnp.asarray(x, jnp.int32)))
    return [LayerStats(*(float(v) if v.ndim == 0 else np.asarray(v, np.float64) for v in s)) for s in st]


# ------------------------------------------------------------------------------------------------- quantisation

def qrow(w):
    """Symmetric int8 per row (last axis) -> (int, scales)."""
    s = np.maximum(np.abs(w).max(-1, keepdims=True), SCALE_FLOOR) / INT8_MAX
    return np.rint(w / s).astype(np.int64), s[..., 0]


def qten(w):
    """Symmetric int8 per tensor -> (int, scale)."""
    s = max(np.abs(w).max(), SCALE_FLOOR) / INT8_MAX
    return np.rint(w / s).astype(np.int64), s


def sat16(x):
    return np.clip(np.rint(x), -32768, 32767)


@dataclasses.dataclass
class Factored:
    """Factored input of a routed layer: T_r[x] = rint(f32(input_proj_r token_code[x]) f32(su_r token_scale)) + c_r."""
    input_proj: np.ndarray            # (E, n, rank) int8
    input_scale: np.ndarray           # (E,) f32 su_r
    bias: np.ndarray                  # (E, n) int16 c_r
    token_code: np.ndarray            # (V, rank) int8
    token_scale: np.ndarray           # (n,) f32


@dataclasses.dataclass
class QuantLayer:
    """One int8 layer of model.bin. A scale is one f32 (per tensor) or an f32 array (group scales)."""
    token_layer: bool
    n: int
    block: int
    depth: int                        # router depth D; 0 for the token layer
    transition: np.ndarray            # (E, n/b, b, b) A16
    readout: np.ndarray               # (E, R, n)
    readout_scale: np.ndarray         # f32, or (E, R) per expert and row
    unembed: np.ndarray               # (V, R)
    unembed_scale: np.ndarray         # f32, or (V,) per class
    table: np.ndarray = None          # (E, V, n) int16; token layer (V, 1, n), its own row; None when factored
    factored: Factored = None
    router_weight: np.ndarray = None  # (2^D - 1, n) state part of each node test
    router_threshold: np.ndarray = None  # (V, 2^D - 1) f32, the token part folded in
    rms_eps: np.float32 = f32(0)
    state_r2: np.ndarray = None       # (n / STATE_GROUP,) f32 squared relative group scales of the layer below
    layer_input_down: np.ndarray = None  # (R2, n)
    layer_input_down_scale: np.ndarray = None  # (R2,) f32
    layer_input_proj: np.ndarray = None  # (E, n, R2)
    layer_input_scale: np.ndarray = None  # f32, or (n,)


def factor(fl, sc):
    """Factored input of a routed layer at per-coordinate state scale sc (n,)."""
    vt, s_vt = qten(fl.token_embed)
    su = np.maximum(np.abs(fl.input_proj).reshape(len(fl.input_proj), -1).max(-1), SCALE_FLOOR) / INT8_MAX
    return Factored(input_proj=np.rint(fl.input_proj / su[:, None, None]).astype(np.int64), input_scale=su.astype(f32),
                    bias=sat16(fl.bias / sc).astype(np.int16), token_code=vt, token_scale=(s_vt / sc).astype(f32))


def quantize(layers, stats, hmargin=1.0, group=False, factored=False):
    """Int8 layers from calibration stats, one state scale per layer or per STATE_GROUP coordinates (group)."""
    out, rel_prev, s_prev = [], None, 1.0
    for fl, st in zip(layers, stats):
        if group:
            g = np.maximum(np.asarray(st.state_group_max, np.float64),
                           GROUP_SCALE_FLOOR * max(np.max(st.state_group_max), SCALE_FLOOR))
            sg = g * hmargin / INT8_MAX
            sc, s_ref = np.repeat(sg, STATE_GROUP), float(sg.max())
        else:
            s_ref = st.state_max * hmargin / INT8_MAX
            sc = np.full(fl.n, s_ref)
        s_z = max(st.readout_max, SCALE_FLOOR) / INT8_MAX
        if group:  # state scales folded into the readout rows
            readout, s_p = qrow(fl.readout * sc[None, None, :])
            readout_scale = (s_p / s_z).astype(f32)
        else:
            readout, s_p = qten(fl.readout)
            readout_scale = f32(s_p * s_ref / s_z)
        unembed, s_q = (qrow if group else qten)(fl.unembed)
        q = QuantLayer(token_layer=fl.token_layer, n=fl.n, block=fl.block, depth=fl.depth,
                       transition=np.rint(fl.transition * 2.0 ** SHIFT).astype(np.int64),
                       readout=readout, readout_scale=readout_scale, unembed=unembed, unembed_scale=f32(s_q * s_z))
        if fl.token_layer:  # one expert per token: only T_x[x]
            ar = np.arange(fl.table.shape[1])
            q.table = sat16(fl.table[ar, ar] / sc).astype(np.int16)[:, None]
        elif factored:
            q.factored = factor(fl, sc)
        else:
            q.table = np.empty(fl.table.shape, np.int16)
            for j in range(0, len(fl.table), EXPERT_CHUNK):
                q.table[j:j + EXPERT_CHUNK] = sat16(fl.table[j:j + EXPERT_CHUNK] / sc)
        if not fl.token_layer:  # the layer below is quantised at s_prev * rel_prev
            F = fl.router_token_dim
            token_part = fl.router_embed @ fl.router_weight[:, :F].T + fl.router_bias[None, :]  # (V, nodes)
            q.router_weight, sw = qrow(fl.router_weight[:, F:] * rel_prev[None, :])
            q.router_threshold = (-token_part / sw[None, :]).astype(f32)
            q.rms_eps = f32(RMS_EPS / s_prev ** 2)
            if group:
                q.state_r2 = (rel_prev[::STATE_GROUP] ** 2).astype(f32)
            if fl.layer_input_down is not None:
                q.layer_input_down, sv = qrow(fl.layer_input_down * rel_prev[None, :])
                s_v = max(st.layer_input_max, SCALE_FLOOR) / INT8_MAX
                q.layer_input_down_scale = (sv / s_v).astype(f32)
                q.layer_input_proj, s_u = qten(fl.layer_input_proj)
                scale = s_u * s_v / sc
                q.layer_input_scale = scale.astype(f32) if group else f32(scale[0])
        rel_prev, s_prev = sc / s_ref, s_ref
        out.append(q)
    return out


def int_forward(layers, output_bias, x):
    """Bit-exact NumPy model of ygg.cpp; x (S, T) -> logits float32 (S, T, V)."""
    S, T = x.shape
    qs = [np.zeros((S, q.n), np.int64) for q in layers]
    out = np.zeros((S, T, len(output_bias)), f32)
    ob = np.asarray(output_bias, f32)
    for t in range(T):
        xt = x[:, t]
        acc = np.broadcast_to(ob, (S, len(ob))).copy()
        for l, q in enumerate(layers):
            if q.token_layer:
                r, trow = xt, np.zeros_like(xt)
            else:
                qb = qs[l - 1]
                if q.state_r2 is not None:  # f32 sum over groups (in order) of r_g^2 times the exact integer S_g
                    Sg = (qb * qb).reshape(S, -1, STATE_GROUP).sum(-1)
                    ssq = np.zeros(S, f32)
                    for g in range(Sg.shape[1]):
                        ssq = (ssq + q.state_r2[g] * Sg[:, g].astype(f32)).astype(f32)
                    rho = f32(1) / np.sqrt(ssq * f32(1.0 / qb.shape[-1]) + q.rms_eps)
                else:
                    rho = f32(1) / np.sqrt((qb * qb).sum(-1).astype(f32) * f32(1.0 / qb.shape[-1]) + q.rms_eps)
                node = np.ones(S, np.int64)  # heap order, 1-based
                for _ in range(q.depth):
                    v = np.einsum("sn,sn->s", q.router_weight[node - 1], qb)
                    node = 2 * node + (v.astype(f32) * rho > q.router_threshold[xt, node - 1])
                r, trow = node - 2 ** q.depth, xt
            b = q.block
            p = (np.einsum("sgij,sgj->sgi", q.transition[r], qs[l].reshape(S, -1, b)).reshape(S, -1)
                 + (1 << (SHIFT - 1))) >> SHIFT
            if q.factored is not None:
                fc = q.factored
                dot = np.einsum("snk,sk->sn", fc.input_proj[r], fc.token_code[xt])
                mul = (fc.input_scale[r][:, None] * fc.token_scale[None, :]).astype(f32)
                drive = p + np.rint(dot.astype(f32) * mul).astype(np.int64) + fc.bias[r].astype(np.int64)
            else:
                drive = p + q.table[r, trow].astype(np.int64)
            if q.layer_input_down is not None:
                zf = (qb @ q.layer_input_down.T).astype(f32) * q.layer_input_down_scale * rho[:, None]
                zl = np.clip(np.rint(zf), -INT8_MAX, INT8_MAX).astype(np.int64)
                ui = np.einsum("skr,sr->sk", q.layer_input_proj[r], zl)
                drive = drive + np.rint(ui.astype(f32) * q.layer_input_scale).astype(np.int64)
            qs[l] = qn = np.clip(drive, -128, 127)
            z = np.einsum("srk,sk->sr", q.readout[r], qn)
            mz = q.readout_scale[r] if np.ndim(q.readout_scale) else q.readout_scale
            zq = np.clip(np.rint(z.astype(f32) * mz), -INT8_MAX, INT8_MAX).astype(np.int64)
            acc = acc + (zq @ q.unembed.T).astype(f32) * q.unembed_scale
        out[:, t] = acc
    return out


# ------------------------------------------------------------------------------------------------- file format

def pack_transition(A16):
    """(E, n/b, b, b) -> per expert [n/16 groups of 16 rows][b/2 column pairs][16 rows][2] int16."""
    E, g, b, _ = A16.shape
    n = g * b
    A = A16.reshape(E, n // 16, 16, b // 2, 2).transpose(0, 1, 3, 2, 4)
    return np.ascontiguousarray(A.astype(np.int16)).reshape(E, -1).view(np.uint8)


def pack_proj(u):
    """(E, n, rank) int8 -> per expert [rank/4][n][4] u8 (u + 128)."""
    E, n, R = u.shape
    return np.ascontiguousarray((u + 128).astype(np.uint8).reshape(E, n, R // 4, 4).transpose(0, 2, 1, 3)).reshape(E, -1)


def pack_readout(pq):
    """(E, R, n) -> per expert u8 (p + 128): R = 8 [n/8][16 lanes][4], R >= 16 [R/16][n/4][16][4]."""
    E, R, n = pq.shape
    if R >= 16:
        a = pq.reshape(E, R // 16, 16, n // 4, 4).transpose(0, 1, 3, 2, 4)
    else:
        a = pq.reshape(E, 8, n // 8, 2, 4).transpose(0, 2, 1, 3, 4)
    return np.ascontiguousarray((a + 128).astype(np.uint8)).reshape(E, -1)


def pack_unembed(Qq, VP):
    """(V, R) -> [R/4][VP][4] u8 (q + 128; padded classes weight 0)."""
    V, R = Qq.shape
    a = np.zeros((VP, R), np.int64)
    a[:V] = Qq
    return np.ascontiguousarray((a + 128).reshape(VP, R // 4, 4).transpose(1, 0, 2).astype(np.uint8))


def header(layers, vocab, router_token_dim=0, input_rank=0):
    """HEADER values of layers, checked against LIMITS (ValueError names the failing field)."""
    q0, routed = layers[0], layers[1:]
    R2 = routed[0].layer_input_proj.shape[2] if routed and routed[0].layer_input_proj is not None else 0
    fac = [q.factored for q in layers if q.factored is not None]
    hdr = dict(n_layers=len(layers), d_state=q0.n, block_size=q0.block, vocab=vocab,
               router_depth=routed[0].depth if routed else 0, readout_rank=q0.readout.shape[1],
               layer_input_rank=R2, reserved=0, router_token_dim=router_token_dim,
               input_rank=fac[0].input_proj.shape[2] if fac else input_rank, shift=SHIFT)

    def need(ok, field):
        if not ok:
            raise ValueError(f"model.bin cannot hold {field} = {hdr.get(field)} (limits {LIMITS})")
    for k in ("d_state", "block_size", "readout_rank"):
        need(hdr[k] in LIMITS[k], k)
    need(R2 % RANK_MULTIPLE == 0 and R2 <= LIMITS["max_layer_input_rank"], "layer_input_rank")
    need(vocab <= LIMITS["max_vocab"], "vocab")
    need(1 <= len(layers) <= LIMITS["max_layers"], "n_layers")
    need(not routed or 1 <= hdr["router_depth"] <= LIMITS["max_router_depth"], "router_depth")
    need(not fac or (hdr["input_rank"] % RANK_MULTIPLE == 0 and RANK_MULTIPLE <= hdr["input_rank"]
                     <= LIMITS["max_input_rank"]), "input_rank")
    for q in routed:
        need(q.depth == hdr["router_depth"], "router_depth")
        need(q.readout.shape[1] == hdr["readout_rank"], "readout_rank")
        need((q.layer_input_proj is not None) == bool(R2) and (not R2 or q.layer_input_proj.shape[2] == R2),
             "layer_input_rank")
    return hdr


def layer_arrays(l, q, hdr):
    """(name, array) of layer l in file order."""
    n, V, R, R2 = hdr["d_state"], hdr["vocab"], hdr["readout_rank"], hdr["layer_input_rank"]
    VP, tag, arrs = round_up(V, CLASS_PAD), f"l{l}_", []
    add = lambda name, a: arrs.append((tag + name, np.ascontiguousarray(a)))
    if not q.token_layer:
        E = 2 ** q.depth
        thr = np.zeros((V, E), f32)  # heap order from node 1: slot 0 unused
        thr[:, 1:] = q.router_threshold
        add("thr", thr)
        if q.state_r2 is not None:
            add("r2b", np.asarray(q.state_r2, f32))
        if np.ndim(q.layer_input_scale):
            add("mulv", np.asarray(q.layer_input_scale, f32))
        if q.factored is not None:
            add("vt", q.factored.token_code.astype(np.int8))
            add("mT", np.asarray(q.factored.token_scale, f32))
        w = np.zeros((E, n), np.int64)
        w[1:] = q.router_weight
        add("w", (w + 128).astype(np.uint8))
        if R2:
            RG = round_up(R2, 16) // 16  # 16-lane groups
            vv = np.zeros((16 * RG, n), np.int64)
            vv[:R2] = q.layer_input_down
            add("vl", (vv + 128).reshape(RG, 16, n // 4, 4).transpose(0, 2, 1, 3).astype(np.uint8))
            mv = np.zeros(16 * RG, f32)
            mv[:R2] = q.layer_input_down_scale
            add("mv", mv)
    parts = {"transition": pack_transition(q.transition)}
    if not q.token_layer and R2:
        parts["layer_input_proj"] = pack_proj(q.layer_input_proj)
    parts["readout"] = pack_readout(q.readout)
    if q.factored is not None:
        parts["input_proj"] = pack_proj(q.factored.input_proj)
        parts["bias"] = np.ascontiguousarray(q.factored.bias).view(np.uint8)
        parts["input_scale"] = np.ascontiguousarray(q.factored.input_scale.astype(f32)[:, None]).view(np.uint8)
    else:
        parts["table"] = q.table.reshape(len(q.table), -1).view(np.uint8)
    offset, off = {k: -1 for k in RECORD}, 0
    for k, a in parts.items():
        offset[k] = off
        off += a.shape[1]
    record = np.zeros((len(parts["transition"]), round_up(off, ALIGN)), np.uint8)
    for k, a in parts.items():
        record[:, offset[k]:offset[k] + a.shape[1]] = a
    del parts
    meta = [record.shape[1]] + [offset[k] for k in RECORD]
    add("meta", np.array(meta if q.factored is not None else meta[:-FACTORED_PARTS], np.int32))
    add("rec", record)
    add("Q", pack_unembed(q.unembed, VP))
    if np.ndim(q.unembed_scale):  # padded classes 0
        sv = np.zeros(VP, f32)
        sv[:V] = q.unembed_scale
        add("sqv", sv)
    if np.ndim(q.readout_scale):  # R = 8: kernel lane i holds row i / 2
        add("mzl", np.repeat(q.readout_scale, 2, axis=1) if R == 8 else np.asarray(q.readout_scale, f32))
    scalars = dict(readout_scale=q.readout_scale, unembed_scale=q.unembed_scale, rms_eps=q.rms_eps,
                   layer_input_scale=q.layer_input_scale)
    add("scal", np.array([0 if v is None or np.ndim(v) else v for v in (scalars[k] for k in SCALARS)], f32))
    return arrs


def write_container(path, hdr, arrs):
    """MAGIC, header, array table, 64-byte aligned arrays -> total array bytes."""
    with open(path, "wb") as f:
        f.write(struct.pack(f"<4s{len(HEADER) + 1}i", MAGIC, *(hdr[k] for k in HEADER), len(arrs)))
        off, tab = 0, []
        for name, a in arrs:
            tab.append((name, off, a.nbytes))
            off += round_up(a.nbytes, ALIGN)
        for name, o, nb in tab:
            f.write(struct.pack("<32sqq", name.encode(), o, nb))
        f.write(b"\0" * ((-f.tell()) % ALIGN))
        for (name, a), (_, o, nb) in zip(arrs, tab):
            f.write(a.tobytes())
            f.write(b"\0" * (round_up(nb, ALIGN) - nb))
    return off


def write(layers, output_bias, path, router_token_dim=0, input_rank=0):
    """QuantLayers -> model.bin; returns (header dict, array bytes)."""
    V = len(output_bias)
    hdr = header(layers, V, router_token_dim, input_rank)
    ob = np.zeros(round_up(V, CLASS_PAD), f32)
    ob[:V] = output_bias
    arrs = [(OUTPUT_BIAS, ob)] + [a for l, q in enumerate(layers) for a in layer_arrays(l, q, hdr)]
    return hdr, write_container(path, hdr, arrs)


# ------------------------------------------------------------------------------------------------- random models

@dataclasses.dataclass(frozen=True)
class KernelConfig:
    """Shape of a random int8 model."""
    d_state: int
    block_size: int
    router_depth: int
    vocab: int
    n_layers: int = 2
    readout_rank: int = 8
    layer_input_rank: int = 16
    input_rank: int = 0               # > 0: factored storage of this rank
    group: bool = False               # group scales


def random_model(cfg, seed=1):
    """Random QuantLayers of shape cfg: orthogonal blocks with decays U[0.6, 1], small tables."""
    rng = np.random.default_rng(seed)
    n, b, D, V, R, R2 = cfg.d_state, cfg.block_size, cfg.router_depth, cfg.vocab, cfg.readout_rank, \
        cfg.layer_input_rank
    orth = np.linalg.qr(rng.standard_normal((64, b, b)))[0]  # 64 distinct orthogonal blocks
    out = []
    for l in range(cfg.n_layers):
        E = V if l == 0 else 2 ** D
        A = np.empty((E, n // b, b, b), np.int64)
        for j in range(0, E, 4096):
            m = min(4096, E - j)
            dec = rng.uniform(0.6, 1.0, (m, n // b, b, 1))
            A[j:j + m] = np.rint(2.0 ** SHIFT * dec * orth[rng.integers(0, 64, (m, n // b))])
        q = dict(token_layer=l == 0, n=n, block=b, depth=0 if l == 0 else D, transition=A)
        if l == 0:
            q["table"] = rng.integers(-24, 25, (V, 1, n), dtype=np.int16)
        else:
            if cfg.input_rank:
                k = cfg.input_rank
                q["factored"] = Factored(
                    input_proj=rng.integers(-127, 128, (E, n, k), dtype=np.int8).astype(np.int64),
                    input_scale=rng.uniform(0.5, 1.5, E).astype(f32), bias=rng.integers(-24, 25, (E, n), dtype=np.int16),
                    token_code=rng.integers(-127, 128, (V, k)), token_scale=rng.uniform(0.0002, 0.001, n).astype(f32))
            else:
                q["table"] = rng.integers(-24, 25, (E, V, n), dtype=np.int16)
            q["router_weight"] = rng.integers(-127, 128, (E - 1, n))
            q["router_threshold"] = (300 * rng.standard_normal((V, E - 1))).astype(f32)
            q["rms_eps"] = f32(1e-4)
            if R2:
                q["layer_input_down"] = rng.integers(-127, 128, (R2, n))
                q["layer_input_down_scale"] = np.full(R2, 0.05, f32)
                q["layer_input_proj"] = rng.integers(-127, 128, (E, n, R2), dtype=np.int8).astype(np.int64)
                q["layer_input_scale"] = f32(0.002)
        q["readout_scale"], q["unembed_scale"] = f32(0.002), f32(0.001)
        if cfg.group:
            q["readout_scale"] = rng.uniform(0.0005, 0.004, (E, R)).astype(f32)
            q["unembed_scale"] = rng.uniform(0.0005, 0.002, V).astype(f32)
            if l > 0:
                q["state_r2"] = (rng.uniform(0.05, 1.0, n // STATE_GROUP) ** 2).astype(f32)
                if R2:
                    q["layer_input_scale"] = rng.uniform(0.0005, 0.004, n).astype(f32)
        q["readout"] = rng.integers(-127, 128, (E, R, n))
        q["unembed"] = rng.integers(-127, 128, (V, R))
        out.append(QuantLayer(**q))
    return out


# ------------------------------------------------------------------------------------------------- main

def checkpoint_task(npz, task="", seq_len=0):
    """(task, window length): given, else from the training result (.json) next to the checkpoint."""
    js = Path(npz).with_suffix(".json")
    if task and seq_len:
        return task, seq_len
    if not js.exists():
        raise SystemExit(f"--task and --seq_len needed: no {js}")
    cfg = upgrade_config(json.loads(js.read_text())["config"])
    return task or cfg["task"], seq_len or cfg["seq_len"]


def export(npz, out, task="", seq_len=0, data=DATA, calib_windows=128, hmargin=0, storage="table",
           state_scale="auto"):
    """Calibrate on training windows, quantize, write out and out.json; returns the export record."""
    t0 = time.time()
    task, seq_len = checkpoint_task(npz, task, seq_len)
    model, layers, output_bias = hard_layers(npz)
    cx, cy = windows(streams(task, data)[0][CALIB_SKIP_TOKENS:], seq_len, calib_windows)
    stats = calibrate(model, cx)
    cal, fac = {}, storage == "factored"
    modes = {"auto": ("tensor", "group"), "tensor": ("tensor",), "group": ("group",)}[state_scale]
    for mode in modes:
        for hm in ((hmargin,) if hmargin else HMARGIN_GRID):
            q = quantize(layers, stats, hm, mode == "group", fac)
            cal[(mode, hm)] = bpc(int_forward(q, output_bias, cx), cy)
    mode, hm = min(cal, key=cal.get)
    args = dict(hmargin=hm, group=mode == "group", factored=fac)  # quantize(layers, stats, **args) rebuilds the file
    hdr, nbytes = write(quantize(layers, stats, **args), output_bias, out,
                        router_token_dim=layers[-1].router_token_dim, input_rank=layers[0].token_embed.shape[1])
    res = dict(header=hdr, bytes=nbytes, task=task, quantize=args, state_scale=mode, storage=storage,
               calib_seq_len=seq_len, calib_windows=calib_windows,
               calib_bpc={f"{k[0]}_{k[1]}": v for k, v in cal.items()},
               stats=[{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in dataclasses.asdict(s).items()}
                      for s in stats],
               secs=round(time.time() - t0, 1))
    Path(str(out) + ".json").write_text(json.dumps(res))
    return res


def main():
    ap = argparse.ArgumentParser(description="checkpoint (.npz) -> ygg.cpp model.bin")
    ap.add_argument("--npz", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", default="", help="enwik8, hg38 (default: from the checkpoint's .json)")
    ap.add_argument("--seq_len", type=int, default=0, help="calibration window length (default: the training "
                    "seq_len); longer windows calibrate the state range for streaming use")
    ap.add_argument("--data", default=str(DATA), help="data directory (default $YGGDRASIL_DATA, else data/)")
    ap.add_argument("--calib_windows", type=int, default=128, help="training-split windows for the scales")
    ap.add_argument("--hmargin", type=float, default=0, help=f"0: pick from {HMARGIN_GRID} by calibration bits")
    ap.add_argument("--storage", default="table", choices=["table", "factored"],
                    help="table: T_r[x] precomputed (V x n int16 per expert); factored: U_r int8 + c_r per expert")
    ap.add_argument("--state_scale", default="auto", choices=["auto", "tensor", "group"],
                    help="one state scale per layer, one per 16 coordinates, or the better by calibration bits")
    rnd = ap.add_argument_group("--random: random weights of this shape instead of --npz")
    rnd.add_argument("--random", action="store_true")
    for f in dataclasses.fields(KernelConfig):
        if f.type is bool:
            rnd.add_argument(f"--{f.name}", action="store_true")
        else:
            rnd.add_argument(f"--{f.name}", type=int, default=None if f.default is dataclasses.MISSING else f.default)
    a = ap.parse_args()
    t0 = time.time()
    if a.random:
        fields = {f.name: getattr(a, f.name) for f in dataclasses.fields(KernelConfig)}
        if missing := [k for k, v in fields.items() if v is None]:
            ap.error(f"--random needs --{', --'.join(missing)}")
        if a.storage == "factored" and not fields["input_rank"]:
            fields["input_rank"] = fields["d_state"] // 4
        cfg = KernelConfig(**fields)
        hdr, nbytes = write(random_model(cfg), np.zeros(cfg.vocab), a.out)
        print(json.dumps(dict(header=hdr, bytes=nbytes, random=True, secs=round(time.time() - t0, 1))))
        return
    if not a.npz:
        ap.error("--npz or --random")
    print(json.dumps(export(a.npz, a.out, a.task, a.seq_len, a.data, a.calib_windows, a.hmargin, a.storage,
                            a.state_scale)))


if __name__ == "__main__":
    sys.exit(main())
