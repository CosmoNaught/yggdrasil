"""Export a train.py checkpoint (.npz) to model.bin for ygg.cpp. Also a bit-exact NumPy model of the int8 kernel.

    python export.py --npz results/<run>.npz --out model.bin
    python export.py --random n,b,D,V[,layers,out_rank,lower_rank,shared_out] --out model.bin   # random weights

Integer scheme, per layer l (state h_l ~ s_l q_l, q_l int8):
  transition  p = (A16_r q + 2^13) >> 14, A16 = rint(2^14 A_r)
  input       T_r[x] = rint((U_r v(x) + c_r) / s_l), int16
  RMS         rho = 1 / sqrt(mean(q_{l-1}^2) + 1e-6 / s_{l-1}^2)
  walk        right iff f32(dot(wq, q_{l-1})) rho > thr[x][node]
  from below  zl = sat8(rint(f32(vq q_{l-1}) m_v rho)), drive += rint(f32(uq_r zl) m_ul)
  readout     zq = sat8(rint(f32(pq_r q_l) m_z)), logits = head_b + sum_l f32(Qq zq) sq_l
--state_scale group: one state scale per 16 coordinates, readout scales per leaf and row, head scales per class.
--storage factored: branch leaves store uf_r int8 and cq_r int16, T_r[x] = rint(f32(uf_r vt[x]) f32(su_r mT)) + cq_r.
"""
import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # repository root
f32 = np.float32
SH = 14  # fixed-point shift of the block maps


# ------------------------------------------------------------------------------------------------- params and data

def load_params(path):
    """train.py .npz -> {"root": {...}, "branches": [{...}, ...], "head_b": ...}."""
    z, P = np.load(path), {"branches": []}
    for name in z.files:
        s = name.split("/")
        if s[0] == "head_b":
            P["head_b"] = z[name]
        elif s[0] == "root":
            P.setdefault("root", {})[s[1]] = z[name]
        else:
            i = int(s[1])
            while len(P["branches"]) <= i:
                P["branches"].append({})
            P["branches"][i][s[2]] = z[name]
    return P


def load_split(task, split):
    """enwik8: bytes split 90M / 5M / 5M. Other tasks: data/<task>/{train,valid,test}.u8."""
    if task == "enwik8":
        raw = np.frombuffer((ROOT / "data" / "enwik8").read_bytes(), np.uint8)
        return {"train": raw[:90_000_000], "val": raw[90_000_000:95_000_000], "test": raw[95_000_000:]}[split]
    return np.memmap(ROOT / "data" / task / f"{ {'val': 'valid'}.get(split, split)}.u8", np.uint8, "r")


def windows(ids, T, m):
    """m windows of T tokens and next-token targets."""
    m = min(m, (len(ids) - 1) // T)
    ids = np.asarray(ids[:m * T + 1], np.int32)
    return ids[:-1].reshape(m, T), ids[1:].reshape(m, T)


def bpc(logits, y):
    lg = logits.astype(np.float64)
    mx = lg.max(-1, keepdims=True)
    lse = np.log(np.exp(lg - mx).sum(-1)) + mx[..., 0]
    return float((lse - np.take_along_axis(lg, y[..., None], -1)[..., 0]).mean() / np.log(2))


# ------------------------------------------------------------------------------------------------- float hard model

def hard_layers(P):
    """Float64 parameters per layer for hard routing, as in train.forward(mode="hard")."""
    import scipy.linalg
    out = []
    for i, p in enumerate([P["root"]] + list(P["branches"])):
        assert "skew" in p and "d_raw" in p, "branch 'out' models are not supported"
        if i > 0:
            assert p["w_node"].shape[1] == p["tok"].shape[1] + p["c"].shape[1], "branch 'tok' models are not supported"
        sk = np.asarray(p["skew"], np.float64)
        sk = 0.5 * (sk - np.swapaxes(sk, -1, -2))  # (leaves, g, b, b)
        NL, g, b, _ = sk.shape
        ex = np.empty_like(sk)
        for j in range(0, NL, 4096):
            ex[j:j + 4096] = scipy.linalg.expm(sk[j:j + 4096])
        d = np.clip(np.asarray(p["d_raw"], np.float64), -1, 1)
        q = dict(n=p["c"].shape[1], b=b, A=d.reshape(NL, g, b)[..., None] * ex,
                 u_in=np.asarray(p["u_in"], np.float64), v_tok=np.asarray(p["v_tok"], np.float64),
                 c=np.asarray(p["c"], np.float64), P=np.asarray(p["p_out"], np.float64),
                 Q=np.asarray(p["q_out"], np.float64), root=i == 0, D=0)
        if "context" in p:
            q["ctx"] = p["context"].shape[0]
        if i > 0:
            F = p["tok"].shape[1]
            q.update(F=F, tok=np.asarray(p["tok"], np.float64), w=np.asarray(p["w_node"], np.float64),
                     bn=np.asarray(p["b_node"], np.float64), D=(p["w_node"].shape[0] + 1).bit_length() - 1)
        if "v_low" in p:
            q["Vl"], q["Ul"] = np.asarray(p["v_low"], np.float64), np.asarray(p["u_low"], np.float64)
        out.append(q)
    return out


def table(q, r, x):
    """T_r[x] = U_r v(x) + c_r for (S,) leaves / tokens."""
    return np.einsum("skr,sr->sk", q["u_in"][r], q["v_tok"][x]) + q["c"][r]


def context_step(c, x, k, V, NL):
    """c = (c mod V^(k-1)) V + x; leaf = c if NL = V^k, else top bits of c * 2654435761 mod 2^32. Returns (c, leaf)."""
    c = (c % V ** (k - 1)) * V + x
    return c, c if V ** k == NL else ((c * 2654435761) & 0xFFFFFFFF) >> (32 - (NL.bit_length() - 1))


def float_forward(HL, head_b, x, stats=False):
    """x (S, T) -> logits (S, T, V), zero initial state. stats=True returns max |h|, |V_low nu|, |P h| instead."""
    S, T = x.shape
    hs = [np.zeros((S, q["n"])) for q in HL]
    st = [dict(h=0.0, v=0.0, z=0.0, hg=np.zeros(q["n"] // 16), zr=np.zeros(q["P"].shape[1])) for q in HL]
    c = np.zeros(S, np.int64)
    out = None if stats else np.zeros((S, T, len(head_b)))
    for t in range(T):
        xt, low = x[:, t], None
        lg = np.broadcast_to(np.asarray(head_b, np.float64), (S, len(head_b))).copy()
        for l, q in enumerate(HL):
            if q["root"]:
                r = xt
                if "ctx" in q:
                    c, r = context_step(c, xt.astype(np.int64), q["ctx"], len(q["v_tok"]), len(q["A"]))
            else:
                f = np.concatenate([q["tok"][xt], low], -1)
                node = np.zeros(S, np.int64)
                for _ in range(q["D"]):
                    node = 2 * node + 1 + (np.einsum("sf,sf->s", f, q["w"][node]) + q["bn"][node] > 0)
                r = node - (2 ** q["D"] - 1)
            drive = table(q, r, xt)
            if "Vl" in q:
                v = low @ q["Vl"].T
                st[l]["v"] = max(st[l]["v"], np.abs(v).max())
                drive = drive + np.einsum("skr,sr->sk", q["Ul"][r], v)
            b = q["b"]
            h = np.einsum("sgij,sgj->sgi", q["A"][r], hs[l].reshape(S, -1, b)).reshape(S, -1) + drive
            hs[l] = h
            st[l]["h"] = max(st[l]["h"], np.abs(h).max())
            st[l]["hg"] = np.maximum(st[l]["hg"], np.abs(h).reshape(S, -1, 16).max((0, 2)))
            z = np.einsum("srk,sk->sr", q["P"][r if len(q["P"]) > 1 else 0 * r], h)
            st[l]["z"] = max(st[l]["z"], np.abs(z).max())
            st[l]["zr"] = np.maximum(st[l]["zr"], np.abs(z).max(0))
            lg += z @ q["Q"].T
            low = h / np.sqrt(np.mean(h * h, -1, keepdims=True) + 1e-6)
        if not stats:
            out[:, t] = lg
    return st if stats else out


# ------------------------------------------------------------------------------------------------- quantisation

def qrow(w):  # per-row symmetric int8 over the last axis
    s = np.maximum(np.abs(w).max(-1, keepdims=True), 1e-12) / 127
    return np.rint(w / s).astype(np.int64), s[..., 0]


def qten(w):  # per-tensor symmetric int8
    s = max(np.abs(w).max(), 1e-12) / 127
    return np.rint(w / s).astype(np.int64), s


def factor(q, sc):
    """Factored input of a branch layer; sc: per-coordinate state scale (n,)."""
    vt, s_vt = qten(q["v_tok"])
    U = q["u_in"]                                    # (NL, n, rank)
    su = np.maximum(np.abs(U).reshape(len(U), -1).max(-1), 1e-12) / 127
    return dict(uf=np.rint(U / su[:, None, None]).astype(np.int64), su=su.astype(f32),
                cq=np.clip(np.rint(q["c"] / sc), -32768, 32767).astype(np.int16), vt=vt,
                mT=(s_vt / sc).astype(f32))


def quantize(HL, st, hmargin=1.0, group=False, factored=False, zmargin=1.0):
    if group:
        return quantize_group(HL, st, hmargin, factored, zmargin)
    Qs, s_prev = [], 1.0
    for l, q in enumerate(HL):
        s_h = st[l]["h"] * hmargin / 127
        NL, V = len(q["A"]), len(q["v_tok"])
        e = dict(n=q["n"], b=q["b"], D=q["D"], root=q["root"], A16=np.rint(q["A"] * 2.0 ** SH).astype(np.int64))
        if q["root"] and "ctx" not in q:  # one leaf per token: store only T_x[x]
            ar = np.arange(V)
            e["T"] = np.clip(np.rint(table(q, ar, ar) / s_h), -32768, 32767).astype(np.int16)[:, None]  # (V, 1, n)
        elif factored and not q["root"]:
            e.update(factor(q, np.full(q["n"], s_h)))
        else:
            e["T"] = np.empty((NL, V, q["n"]), np.int16)
            for j in range(0, NL, 256):
                Tf = np.einsum("lkr,vr->lvk", q["u_in"][j:j + 256], q["v_tok"]) + q["c"][j:j + 256, None]
                e["T"][j:j + 256] = np.clip(np.rint(Tf / s_h), -32768, 32767)
        if "ctx" in q:
            e["ctx"] = q["ctx"]
        if not q["root"]:
            F = q["F"]
            G = q["tok"] @ q["w"][:, :F].T + q["bn"][None, :]  # (V, nodes): token part of each node test
            e["wq"], sw = qrow(q["w"][:, F:])
            e["thr"] = (-G / sw[None, :]).astype(f32)
            e["eps"] = f32(1e-6 / s_prev ** 2)
            if "Vl" in q:
                e["vq"], sv = qrow(q["Vl"])
                s_v = max(st[l]["v"], 1e-12) / 127
                e["m_v"] = (sv / s_v).astype(f32)
                e["uq"], s_u = qten(q["Ul"])
                e["m_ul"] = f32(s_u * s_v / s_h)
        e["pq"], s_p = qten(q["P"])
        s_z = max(st[l]["z"], 1e-12) / 127
        e["m_z"] = f32(s_p * s_h / s_z)
        e["Qq"], s_q = qten(q["Q"])
        e["sq"] = f32(s_q * s_z)
        s_prev = s_h
        Qs.append(e)
    return Qs


def quantize_group(HL, st, hmargin=1.0, factored=False, zmargin=1.0):
    """quantize() with one state scale per 16 coordinates."""
    Qs, rc_prev, sref_prev = [], None, 1.0
    for l, q in enumerate(HL):
        n = q["n"]
        hg = np.maximum(np.asarray(st[l]["hg"], np.float64), 1e-3 * max(np.max(st[l]["hg"]), 1e-12))
        sg = hg * hmargin / 127
        sc = np.repeat(sg, 16)                      # per-coordinate scale
        sref = float(sg.max())
        NL, V = len(q["A"]), len(q["v_tok"])
        e = dict(n=n, b=q["b"], D=q["D"], root=q["root"], A16=np.rint(q["A"] * 2.0 ** SH).astype(np.int64))
        if q["root"] and "ctx" not in q:
            ar = np.arange(V)
            e["T"] = np.clip(np.rint(table(q, ar, ar) / sc), -32768, 32767).astype(np.int16)[:, None]
        elif factored and not q["root"]:
            e.update(factor(q, sc))
        else:
            e["T"] = np.empty((NL, V, n), np.int16)
            for j in range(0, NL, 256):
                Tf = np.einsum("lkr,vr->lvk", q["u_in"][j:j + 256], q["v_tok"]) + q["c"][j:j + 256, None]
                e["T"][j:j + 256] = np.clip(np.rint(Tf / sc), -32768, 32767)
        if "ctx" in q:
            e["ctx"] = q["ctx"]
        if not q["root"]:
            F = q["F"]
            G = q["tok"] @ q["w"][:, :F].T + q["bn"][None, :]
            e["wq"], sw = qrow(q["w"][:, F:] * rc_prev[None, :])
            e["thr"] = (-G / sw[None, :]).astype(f32)
            e["eps"] = f32(1e-6 / sref_prev ** 2)
            e["r2b"] = (rc_prev[::16] ** 2).astype(f32)
            if "Vl" in q:
                e["vq"], sv = qrow(q["Vl"] * rc_prev[None, :])
                s_v = max(st[l]["v"], 1e-12) / 127
                e["m_v"] = (sv / s_v).astype(f32)
                e["uq"], s_u = qten(q["Ul"])
                e["mulv"] = (s_u * s_v / sc).astype(f32)
                e["m_ul"] = f32(0)
        e["pq"], s_p = qrow(q["P"] * sc[None, None, :])           # (leaves, R) scales
        s_z = max(st[l]["z"], 1e-12) * zmargin / 127              # one z scale
        e["mzl"] = (s_p / s_z).astype(f32)
        e["m_z"] = f32(0)
        e["Qq"], s_qv = qrow(q["Q"])                                # one Q scale per class
        e["sqv"] = (s_qv * s_z).astype(f32)
        e["sq"] = f32(0)
        rc_prev, sref_prev = sc / sref, sref
        Qs.append(e)
    return Qs


def int_forward(Qs, head_b, x):
    """Bit-exact NumPy model of ygg.cpp; x (S, T) -> logits float32 (S, T, V)."""
    S, T = x.shape
    qs = [np.zeros((S, e["n"]), np.int64) for e in Qs]
    out = np.zeros((S, T, len(head_b)), f32)
    hb = np.asarray(head_b, f32)
    c = np.zeros(S, np.int64)
    for t in range(T):
        xt = x[:, t]
        acc = np.broadcast_to(hb, (S, len(hb))).copy()
        for l, e in enumerate(Qs):
            if e["root"] and "ctx" in e:
                c, r = context_step(c, xt.astype(np.int64), e["ctx"], len(head_b), len(e["A16"]))
                trow = xt
            elif e["root"]:
                r, trow = xt, np.zeros_like(xt)
            else:
                qb = qs[l - 1]
                if "r2b" in e:  # group scales: f32 sum over groups (in order) of r_g^2 times the exact integer S_g
                    Sg = (qb * qb).reshape(S, -1, 16).sum(-1)
                    ssq = np.zeros(S, f32)
                    for g in range(Sg.shape[1]):
                        ssq = (ssq + e["r2b"][g] * Sg[:, g].astype(f32)).astype(f32)
                    rho = f32(1) / np.sqrt(ssq * f32(1.0 / qb.shape[-1]) + e["eps"])
                else:
                    rho = f32(1) / np.sqrt((qb * qb).sum(-1).astype(f32) * f32(1.0 / qb.shape[-1]) + e["eps"])
                node = np.ones(S, np.int64)  # Eytzinger, 1-based
                for _ in range(e["D"]):
                    v = np.einsum("sn,sn->s", e["wq"][node - 1], qb)
                    node = 2 * node + (v.astype(f32) * rho > e["thr"][xt, node - 1])
                r, trow = node - 2 ** e["D"], xt
            b = e["b"]
            p = (np.einsum("sgij,sgj->sgi", e["A16"][r], qs[l].reshape(S, -1, b)).reshape(S, -1) + (1 << (SH - 1))) >> SH
            if "uf" in e:  # factored input
                dot = np.einsum("snk,sk->sn", e["uf"][r], e["vt"][xt])
                mul = (e["su"][r][:, None] * e["mT"][None, :]).astype(f32)
                drive = p + np.rint(dot.astype(f32) * mul).astype(np.int64) + e["cq"][r].astype(np.int64)
            else:
                drive = p + e["T"][r, trow].astype(np.int64)
            if "vq" in e:
                zf = (qb @ e["vq"].T).astype(f32) * e["m_v"] * rho[:, None]
                zl = np.clip(np.rint(zf), -127, 127).astype(np.int64)
                ui = np.einsum("skr,sr->sk", e["uq"][r], zl)
                drive = drive + np.rint(ui.astype(f32) * (e["mulv"] if "mulv" in e else e["m_ul"])).astype(np.int64)
            qs[l] = qn = np.clip(drive, -128, 127)
            z = np.einsum("srk,sk->sr", e["pq"][r if len(e["pq"]) > 1 else 0 * r], qn)
            if "mzl" in e:  # per leaf and row
                zq = np.clip(np.rint(z.astype(f32) * e["mzl"][r if len(e["mzl"]) > 1 else 0 * r]), -127, 127).astype(np.int64)
            else:
                zq = np.clip(np.rint(z.astype(f32) * e["m_z"]), -127, 127).astype(np.int64)
            acc = acc + (zq @ e["Qq"].T).astype(f32) * (e["sqv"][None, :] if "sqv" in e else e["sq"])
        out[:, t] = acc
    return out


# ------------------------------------------------------------------------------------------------- file format

def pack_A(A16):
    """(NL, n/b, b, b) -> per leaf [n/16 groups of 16 rows][b/2 column pairs][16 rows][2] int16."""
    NL, g, b, _ = A16.shape
    n = g * b
    A = A16.reshape(NL, n // 16, 16, b // 2, 2).transpose(0, 1, 3, 2, 4)
    return np.ascontiguousarray(A.astype(np.int16)).reshape(NL, -1).view(np.uint8)


def pack_U(uq):  # (NL, n, R2) -> per leaf [R2/4][n][4] u8 = u + 128
    NL, n, R2 = uq.shape
    return np.ascontiguousarray((uq + 128).astype(np.uint8).reshape(NL, n, R2 // 4, 4).transpose(0, 2, 1, 3)).reshape(NL, -1)


def pack_P(pq):  # (NL, R, n) -> per leaf u8 (p + 128): R = 8 [n/8][16 lanes][4], R >= 16 [R/16][n/4][16][4]
    NL, R, n = pq.shape
    if R >= 16:
        a = pq.reshape(NL, R // 16, 16, n // 4, 4).transpose(0, 1, 3, 2, 4)
    else:
        a = pq.reshape(NL, 8, n // 8, 2, 4).transpose(0, 2, 1, 3, 4)
    return np.ascontiguousarray((a + 128).astype(np.uint8)).reshape(NL, -1)


def pack_Q(Qq, VP):  # (V, R) -> [R/4][VP][4] u8 (q + 128; pad classes weight 0)
    V, R = Qq.shape
    a = np.zeros((VP, R), np.int64)
    a[:V] = Qq
    return np.ascontiguousarray((a + 128).reshape(VP, R // 4, 4).transpose(1, 0, 2).astype(np.uint8))


def write(Qs, head_b, path, info):
    """Header "YGG1" + int32 [L, n, b, V, D, out_rank, lower_rank, shared_out, feat, rank, sh, n_arrays], a table of
    (name[32], offset, bytes), then 64-byte aligned arrays."""
    arrs = []
    add = lambda name, a: arrs.append((name, np.ascontiguousarray(a)))
    L, n, b = len(Qs), Qs[0]["n"], Qs[0]["b"]
    V = len(head_b)
    VP = (V + 31) // 32 * 32
    R = Qs[0]["pq"].shape[1]
    R2 = Qs[1]["uq"].shape[2] if L > 1 and "uq" in Qs[1] else 0
    D = Qs[1]["D"] if L > 1 else 0
    shared = int(len(Qs[0]["pq"]) == 1)
    assert n in (64, 128, 256) and b in (8, 16) and R in (8, 16, 32) and R2 % 4 == 0 and R2 <= 64 and V <= 256 and L <= 8
    for e in Qs[1:]:
        assert e["D"] == D and e["pq"].shape[1] == R and (len(e["pq"]) == 1) == bool(shared)
        assert ("uq" in e) == bool(R2) and (not R2 or e["uq"].shape[2] == R2)
    RG = max(1, (R2 + 15) // 16)  # 16-lane groups of the low-rank input
    fac = [e for e in Qs if "uf" in e]
    rank = fac[0]["uf"].shape[2] if fac else info.get("rank", 0)
    assert not fac or (rank % 4 == 0 and rank <= 64)
    hdr = [L, n, b, V, D, R, R2, shared, info.get("feat", 0), rank, SH]
    hb = np.zeros(VP, f32)
    hb[:V] = head_b
    add("head_b", hb)
    for l, e in enumerate(Qs):
        P = f"l{l}_"
        if "ctx" in e:  # context root: k, V^(k - 1), hash bits (0: exact index)
            NL, k = len(e["A16"]), e["ctx"]
            add(P + "ctx", np.array([k, V ** (k - 1), 0 if V ** k == NL else NL.bit_length() - 1], np.int32))
        if not e["root"]:
            NL = 2 ** D
            thr = np.zeros((V, NL), f32)
            thr[:, 1:] = e["thr"]
            add(P + "thr", thr)
            if "r2b" in e:
                add(P + "r2b", np.asarray(e["r2b"], f32))
            if "mulv" in e:
                add(P + "mulv", np.asarray(e["mulv"], f32))
            if "uf" in e:
                add(P + "vt", e["vt"].astype(np.int8))
                add(P + "mT", np.asarray(e["mT"], f32))
            w = np.zeros((NL, n), np.int64)
            w[1:] = e["wq"]
            add(P + "w", (w + 128).astype(np.uint8))
            if R2:
                vv = np.zeros((16 * RG, n), np.int64)
                vv[:R2] = e["vq"]
                add(P + "vl", (vv + 128).reshape(RG, 16, n // 4, 4).transpose(0, 2, 1, 3).astype(np.uint8))
                mv = np.zeros(16 * RG, f32)
                mv[:R2] = e["m_v"]
                add(P + "mv", mv)
        parts = [("A", pack_A(e["A16"]))]
        if not e["root"] and R2:
            parts.append(("U", pack_U(e["uq"])))
        if shared:
            add(P + "P", pack_P(e["pq"]))
        else:
            parts.append(("P", pack_P(e["pq"])))
        if "uf" in e:  # factored: U_r [rank/4][n][4] u8, c_r int16, su_r f32
            parts.append(("F", pack_U(e["uf"])))
            parts.append(("C", np.ascontiguousarray(e["cq"]).view(np.uint8)))
            parts.append(("S", np.ascontiguousarray(e["su"].astype(f32)[:, None]).view(np.uint8)))
        else:
            parts.append(("T", e["T"].reshape(len(e["T"]), -1).view(np.uint8)))
        meta, off = {"A": -1, "U": -1, "P": -1, "T": -1, "F": -1, "C": -1, "S": -1}, 0
        for nm, a in parts:
            meta[nm] = off
            off += a.shape[1]
        RS = (off + 63) // 64 * 64
        rec = np.zeros((len(parts[0][1]), RS), np.uint8)
        for nm, a in parts:
            rec[:, meta[nm]:meta[nm] + a.shape[1]] = a
        del parts
        mt = [RS, meta["A"], meta["U"], meta["P"], meta["T"]] + ([meta["F"], meta["C"], meta["S"]] if "uf" in e else [])
        add(P + "meta", np.array(mt, np.int32))
        add(P + "rec", rec)
        add(P + "Q", pack_Q(e["Qq"], VP))
        if "sqv" in e:  # class scales of the head (padded classes 0)
            sv = np.zeros(VP, f32)
            sv[:V] = e["sqv"]
            add(P + "sqv", sv)
        if "mzl" in e:  # R = 8: kernel lane i holds row i / 2
            add(P + "mzl", np.repeat(e["mzl"], 2, axis=1) if R == 8 else np.asarray(e["mzl"], f32))
        add(P + "scal", np.array([e["m_z"], e["sq"], e.get("eps", 0), e.get("m_ul", 0)], f32))
    with open(path, "wb") as f:
        f.write(struct.pack("<4s12i", b"YGG1", *hdr, len(arrs)))
        off, tab = 0, []
        for name, a in arrs:
            tab.append((name, off, a.nbytes))
            off += (a.nbytes + 63) // 64 * 64
        for name, o, nb in tab:
            f.write(struct.pack("<32sqq", name.encode(), o, nb))
        f.write(b"\0" * ((-f.tell()) % 64))
        for (name, a), (_, o, nb) in zip(arrs, tab):
            f.write(a.tobytes())
            f.write(b"\0" * ((nb + 63) // 64 * 64 - nb))
    return dict(zip(["L", "n", "b", "V", "D", "out_rank", "lower_rank", "shared_out", "feat", "rank", "sh"], hdr)), off


# ------------------------------------------------------------------------------------------------- random models

def random_quantized(n, b, D, V, L=2, R=8, R2=16, shared=0, seed=1, group=False, factored=0):
    """Random int8 model in the export format: orthogonal blocks with decays U[0.6, 1], small tables."""
    rng = np.random.default_rng(seed)
    g = n // b
    orth = np.linalg.qr(rng.standard_normal((64, b, b)))[0]  # 64 distinct orthogonal blocks
    Qs = []
    for l in range(L):
        NL = V if l == 0 else 2 ** D
        A = np.empty((NL, g, b, b), np.int64)
        for j in range(0, NL, 4096):
            m = min(4096, NL - j)
            dec = rng.uniform(0.6, 1.0, (m, g, b, 1))
            A[j:j + m] = np.rint(2.0 ** SH * dec * orth[rng.integers(0, 64, (m, g))])
        e = dict(n=n, b=b, D=0 if l == 0 else D, root=l == 0, A16=A)
        if l == 0:
            e["T"] = rng.integers(-24, 25, (V, 1, n), dtype=np.int16)
        else:
            if factored:  # factored input of rank `factored`
                e["uf"] = rng.integers(-127, 128, (NL, n, factored), dtype=np.int8).astype(np.int64)
                e["su"] = rng.uniform(0.5, 1.5, NL).astype(f32)
                e["cq"] = rng.integers(-24, 25, (NL, n), dtype=np.int16)
                e["vt"] = rng.integers(-127, 128, (V, factored))
                e["mT"] = rng.uniform(0.0002, 0.001, n).astype(f32)
            else:
                e["T"] = rng.integers(-24, 25, (NL, V, n), dtype=np.int16)
            e["wq"] = rng.integers(-127, 128, (NL - 1, n))
            e["thr"] = (300 * rng.standard_normal((V, NL - 1))).astype(f32)
            e["eps"] = f32(1e-4)
            if R2:
                e["vq"] = rng.integers(-127, 128, (R2, n))
                e["m_v"] = np.full(R2, 0.05, f32)
                e["uq"] = rng.integers(-127, 128, (NL, n, R2), dtype=np.int8).astype(np.int64)
                e["m_ul"] = f32(0.002)
        if group:  # readout scale per leaf and row, head scale per class
            e["mzl"] = rng.uniform(0.0005, 0.004, (1 if shared else NL, R)).astype(f32)
            e["sqv"] = rng.uniform(0.0005, 0.002, V).astype(f32)
        if group and l > 0:  # group scales: r^2 of the previous layer, m_ul per coordinate
            e["r2b"] = (rng.uniform(0.05, 1.0, n // 16) ** 2).astype(f32)
            if R2:
                e["mulv"] = rng.uniform(0.0005, 0.004, n).astype(f32)
        e["pq"] = rng.integers(-127, 128, (1 if shared else NL, R, n))
        e["m_z"] = f32(0.002)
        e["Qq"] = rng.integers(-127, 128, (V, R))
        e["sq"] = f32(0.001)
        Qs.append(e)
    return Qs


# ------------------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="models/yggdrasil/training/train.py .npz -> ygg.cpp model.bin")
    ap.add_argument("--npz", default="")
    ap.add_argument("--random", default="", help="n,b,D,V[,layers,out_rank,lower_rank,shared_out]: random weights")
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", default="", help="enwik8 / hg38 (default: from the checkpoint's .json)")
    ap.add_argument("--calib_windows", type=int, default=128, help="training-split windows for the scales")
    ap.add_argument("--calib_T", type=int, default=0, help="calibration window length (0: the checkpoint's T); longer "
                    "windows calibrate the state range for streaming use")
    ap.add_argument("--hmargin", type=float, default=0, help="0: pick from {1, .85, .7, .6} by calibration bpc")
    ap.add_argument("--storage", default="table", choices=["table", "factored"],
                    help="table: T_r[x] precomputed (V x n int16 per leaf); factored: U_r int8 + c_r per leaf (1 B/param)")
    ap.add_argument("--state_scale", default="auto", choices=["auto", "tensor", "group"],
                    help="one state scale per layer, one per 16 coordinates, or the better by calibration bpc")
    a = ap.parse_args()
    t0 = time.time()
    if a.random:
        sp = [int(v) for v in a.random.split(",")]
        sp += [2, 8, 16, 0, 0][len(sp) - 4:]
        n, b, D, V, L, R, R2, shared, frank = sp[:9]
        frank = frank or (n // 4 if a.storage == "factored" else 0)
        hdr, nbytes = write(random_quantized(n, b, D, V, L, R, R2, shared, factored=frank), np.zeros(V), a.out, {})
        print(json.dumps(dict(hdr=hdr, bytes=nbytes, random=True, secs=round(time.time() - t0, 1))))
        return
    cfg = json.loads(Path(a.npz).with_suffix(".json").read_text())["config"]
    task = a.task or cfg["task"]
    P = load_params(a.npz)
    HL = hard_layers(P)
    cx, cy = windows(load_split(task, "train")[1_000_000:], a.calib_T or cfg["T"], a.calib_windows)
    st = float_forward(HL, P["head_b"], cx, stats=True)
    cal, fac = {}, a.storage == "factored"
    modes = {"auto": ("tensor", "group"), "tensor": ("tensor",), "group": ("group",)}[a.state_scale]
    for mode in modes:
        for hm in ((a.hmargin,) if a.hmargin else (1.0, 0.85, 0.7, 0.6)):
            cal[(mode, hm)] = bpc(int_forward(quantize(HL, st, hm, mode == "group", fac), P["head_b"], cx), cy)
    mode, hm = min(cal, key=cal.get)
    Qs = quantize(HL, st, hm, mode == "group", fac)
    hdr, nbytes = write(Qs, P["head_b"], a.out, dict(feat=cfg.get("feat", 0), rank=cfg.get("rank", 0)))
    res = dict(hdr=hdr, bytes=nbytes, task=task, hmargin=hm, state_scale=mode, storage=a.storage,
               calib_T=a.calib_T or cfg["T"],
               calib_windows=a.calib_windows, calib_bpc={f"{k[0]}_{k[1]}": v for k, v in cal.items()},
               stats=[{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()} for d in st],
               secs=round(time.time() - t0, 1))
    Path(a.out + ".json").write_text(json.dumps(res))
    print(json.dumps(res))


if __name__ == "__main__":
    sys.exit(main())
