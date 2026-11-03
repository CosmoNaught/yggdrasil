"""Yggdrasil: a recurrent layer that retrieves its transition from a tree of stored leaves.

    h^l_t  = A_r h^l_{t-1} + T_r[x_t] + U_r W nu(h^{l-1}_t)
    A_r    = diag(clip(d_r, -1, 1)) blockdiag(expm(S_r / 2 - S_r^T / 2))
    logits = beta + sum_l Q^l P^l_r h^l_t

    python models/yggdrasil/training/train.py train --task s5_word
    python models/yggdrasil/training/train.py train --task enwik8 --depth 12
    python models/yggdrasil/training/train.py train --task human_nontata_promoters
"""

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))  # kernel modules
import leaves  # noqa: E402
LN2 = float(np.log(2.0))
SKEW_STD = 0.05  # skew init std of the first block; the last block has std 1
ROUTE_TAU = 0.5  # straight-through routing temperature
LEAF_MM = "grouped"  # large-vocabulary branches: "grouped" leaf matmuls, or "dense" over all leaves
LM_KERNEL = True  # large-vocabulary form on GPU: blocked scan kernel reading A from the leaf table (leaf_scan.py)
LM_DTYPE = jnp.bfloat16  # large-vocabulary form: operand type of the leaf and head matmuls (fp32 accumulation)
SCAN_IMPL = "scan"  # large-vocabulary form on GPU: "scan" (scan.py) or "leaf_scan" (leaf_scan.py)
MESH = None  # large-vocabulary form: a 1-D device mesh (axis "x") shards the branch leaves by leaf id


# ----------------------------------------------------------------------------------------------------------- model

def _normal(key, shape, scale):
    return scale * jax.random.normal(key, shape)


def init(key, vocab, classes, n=128, block=16, depth=8, layers=2, feat=16, rank=16, out_rank=8, lower_rank=16,
         shared_out=False, branch="dyn", context=1, bits=16, emb=0, root_rank=0, branch_leaves=None):
    """branch: "dyn" (default), "out" (no stored transition, A_r = 0), "tok" (tree reads the token only).

    emb > 0 (large vocabularies): one token embedding E (vocab, emb) shared by all layers and tied to the head,
    logits = beta + E g rms(sum_l Q^l P^l_r h^l), Q^l (emb, out_rank); branches read v(x) = rms(E x) W_v and route on
    [rms(E x); norm(h^{l-1})]. root_rank > 0: root leaf x = base leaf + (rms(E x) R) G, a rank root_rank leaf table.
    branch_leaves: draw only this many branch leaves (to_tree keeps leaf 0 only).
    """
    assert n % block == 0 and branch in ("dyn", "out", "tok") and (branch != "tok" or feat != n)
    assert not root_rank or (emb and context == 1)
    feat = emb or feat
    assert context == 1 or vocab**context < 2**32
    g = n // block
    keys = iter(jax.random.split(key, 16 * layers + 2))
    parts = []
    for i in range(layers):
        leaves = (1 if root_rank else vocab if context == 1 else min(vocab**context, 2**bits)) if i == 0 else \
            branch_leaves or 2**depth
        tok = _normal(next(keys), (vocab, feat), 1.0)  # drawn for the root too to keep the key order
        p = {"v_tok": _normal(next(keys), (vocab, rank), 1.0),
             "u_in": _normal(next(keys), (leaves, n, rank), 0.5 * rank**-0.5),
             "c": _normal(next(keys), (leaves, n), 0.1),
             "p_out": _normal(next(keys), (leaves, out_rank, n), n**-0.5),
             "q_out": _normal(next(keys), (classes, out_rank), 0.1 * out_rank**-0.5)}
        if shared_out:
            p["p_out"] = p["p_out"][:1]
        if i > 0:
            n_in = feat if branch == "tok" else feat + n
            p["tok"] = tok
            p["w_node"] = _normal(next(keys), (2**depth - 1, n_in), n_in**-0.5)
            p["b_node"] = jnp.zeros((2**depth - 1,))
        stds = SKEW_STD * (1.0 / SKEW_STD) ** (jnp.arange(g) / max(1, g - 1))
        p["skew"] = _normal(next(keys), (leaves, g, block, block), SKEW_STD) * (stds / SKEW_STD)[:, None, None]
        p["d_raw"] = jax.random.uniform(next(keys), (leaves, n), minval=0.6, maxval=1.0).at[:, :block].set(0.0)
        if branch == "out" and i > 0:
            del p["skew"], p["d_raw"]
        if lower_rank and i > 0:
            p["v_low"] = _normal(next(keys), (lower_rank, n), n**-0.5)
            p["u_low"] = jnp.zeros((leaves, n, lower_rank))
        if context > 1 and i == 0:
            p["context"] = jnp.zeros((context, 0))  # stores the context length as a shape
        parts.append(p)
    out = {"root": parts[0], "branches": parts[1:], "head_b": jnp.zeros((classes,))}
    if emb:
        assert classes == vocab
        ek = iter(jax.random.split(jax.random.fold_in(key, 1), 4 * layers + 2))
        out["emb"], out["out_g"] = _normal(next(ek), (vocab, emb), 0.02), jnp.ones((emb,))
        for i, p in enumerate(parts):
            del p["v_tok"]
            p["q_out"] = _normal(next(ek), (emb, out_rank), out_rank**-0.5)
            if i > 0:
                del p["tok"]
                p["w_v"] = _normal(next(ek), (emb, rank), emb**-0.5)
        if root_rank:
            p = parts[0]
            del p["u_in"]
            stds = SKEW_STD * (1.0 / SKEW_STD) ** (jnp.arange(g) / max(1, g - 1))
            p["r_in"] = _normal(next(ek), (emb, root_rank), emb**-0.5)
            p["g_c"] = _normal(next(ek), (root_rank, n), 0.5 * root_rank**-0.5)
            p["g_d"] = _normal(next(ek), (root_rank, n), 0.1 * root_rank**-0.5).at[:, :block].set(0.0)
            p["g_s"] = (_normal(next(ek), (root_rank, g, block, block), 0.5 * root_rank**-0.5)
                        * stds[:, None, None]).reshape(root_rank, -1)
    return out


def context_leaf(xs, k, vocab, leaves, prev=None):
    """Root leaf ids (T, B) from the last k tokens; hashed to 2^bits leaves when vocab^k > leaves."""
    pad = jnp.zeros((k - 1, xs.shape[1]), xs.dtype) if prev is None else jnp.swapaxes(prev, 0, 1).astype(xs.dtype)
    full = jnp.concatenate([pad, xs]).astype(jnp.uint32)
    c = jnp.zeros(xs.shape, jnp.uint32)
    for j in range(k):
        c = c * jnp.uint32(vocab) + full[j:j + xs.shape[0]]
    if vocab**k == leaves:
        return c.astype(jnp.int32)
    return ((c * jnp.uint32(2654435761)) >> jnp.uint32(32 - (leaves.bit_length() - 1))).astype(jnp.int32)


def _expm(sk, chunk=16384):
    """expm of (..., b, b), in checkpointed chunks when large."""
    flat = sk.reshape((-1,) + sk.shape[-2:])
    if flat.shape[0] <= chunk:
        return jax.scipy.linalg.expm(sk)
    pad = -flat.shape[0] % chunk
    flat = jnp.concatenate([flat, jnp.zeros((pad,) + flat.shape[1:], flat.dtype)]) if pad else flat
    out = jax.lax.map(jax.checkpoint(jax.scipy.linalg.expm), flat.reshape((-1, chunk) + flat.shape[1:]))
    return out.reshape((-1,) + flat.shape[1:])[:flat.shape[0] - pad].reshape(sk.shape)


def _expm_skew(sk, squarings=8, order=8):
    """expm of skew-symmetric (..., b, b): Taylor of the given order after 2^squarings scaling, then squaring."""
    x = sk / 2.0**squarings
    eye = jnp.eye(sk.shape[-1], dtype=sk.dtype)
    r = eye + x / order
    for k in range(order - 1, 0, -1):
        r = eye + x @ r / k
    for _ in range(squarings):
        r = r @ r
    return r


def transitions(p):
    """(leaves, n / b, b, b)"""
    sk = 0.5 * (p["skew"] - jnp.swapaxes(p["skew"], -1, -2))
    return jnp.clip(p["d_raw"], -1.0, 1.0).reshape(sk.shape[:-1])[..., None] * _expm(sk)


def _scan(a, b):
    """h_t = a_t h_{t-1} + b_t, h_0 = 0: a (T, ..., b, b), b (T, ..., b)."""
    comb = lambda l, r: (jnp.einsum("...ij,...jk->...ik", r[0], l[0]),
                         jnp.einsum("...ij,...j->...i", r[0], l[1]) + r[1])
    return jax.lax.associative_scan(comb, (a, b), axis=0)[1]


def _rms(h):
    return h * jax.lax.rsqrt(jnp.mean(jnp.square(h), axis=-1, keepdims=True) + 1e-6)


def walk(p, f, beam, tau, mode, key=None, route_noise=0.0, dense=False):
    """f (T, B, F) -> leaf ids (T, B, K), weights (T, B, K). Slot 0 is the greedy path.
    dense: score every node once, f W^T (T, B, 2^D - 1), then pick the walked ones."""
    depth = (p["w_node"].shape[0] + 1).bit_length() - 1
    beam = 1 if mode == "hard" else beam
    node = jnp.zeros(f.shape[:2] + (1,), jnp.int32)
    lp = jnp.zeros(f.shape[:2] + (1,), f.dtype)
    scores = _mm(f, p["w_node"].T) + p["b_node"] if dense else None
    for level in range(depth):
        if dense:
            s = jnp.take_along_axis(scores, node, -1)
        else:
            s = jnp.einsum("tbf,tbqf->tbq", f, p["w_node"][node]) + p["b_node"][node]
        if key is not None and route_noise > 0:
            s = s + route_noise * jax.random.logistic(jax.random.fold_in(key, level), s.shape)
        cand_lp = jnp.stack([lp + jax.nn.log_sigmoid(-s / tau), lp + jax.nn.log_sigmoid(s / tau)], -1)
        cand_lp = cand_lp.reshape(s.shape[:-1] + (-1,))
        cand = jnp.stack([2 * node + 1, 2 * node + 2], -1).reshape(cand_lp.shape)
        greedy = jax.nn.one_hot((s[..., 0] > 0).astype(jnp.int32), cand.shape[-1], dtype=f.dtype)
        _, sel = jax.lax.top_k(jax.lax.stop_gradient(cand_lp) + 1e9 * greedy, min(cand.shape[-1], beam))
        node, lp = jnp.take_along_axis(cand, sel, -1), jnp.take_along_axis(cand_lp, sel, -1)
    leaf = node - (2**depth - 1)
    if mode == "hard":
        return leaf, jnp.ones_like(lp)
    w = jax.nn.softmax(lp, -1)
    return leaf, jax.nn.one_hot(jnp.zeros(lp.shape[:-1], jnp.int32), lp.shape[-1], dtype=w.dtype) + w - \
        jax.lax.stop_gradient(w)


def part(p, leaf, w, xs, low, noise, h0=None):
    """leaf, w (T, B, K), low (T, B, n) or None, h0 (B, n) or None -> states (T, B, n)."""
    n_leaves, n = p["c"].shape
    vocab = p["v_tok"].shape[0]
    if n_leaves * vocab * n <= 2**28:  # T_r[x] = U_r v(x) + c_r
        table = jnp.einsum("lkr,vr->lvk", p["u_in"], p["v_tok"]) + p["c"][:, None, :]
        drive = jnp.einsum("tbq,tbqk->tbk", w, table.reshape(-1, n)[leaf * vocab + xs[:, :, None]])
    else:
        drive = jnp.einsum("tbq,tbqkr,tbr->tbk", w, p["u_in"][leaf], p["v_tok"][xs]) + \
            jnp.einsum("tbq,tbqk->tbk", w, p["c"][leaf])
    if "v_low" in p:
        drive = drive + jnp.einsum("tbq,tbqkr,tbr->tbk", w, p["u_low"][leaf], low @ p["v_low"].T)
    if noise is not None:
        drive = drive + noise
    if "skew" not in p:  # branch "out"
        return drive
    abar = jnp.einsum("tbq,tbq...->tb...", w, transitions(p)[leaf])
    shape, drive = drive.shape, drive.reshape(abar.shape[:-1])
    if h0 is not None:  # fold A_1 h0 into the first drive
        drive = drive.at[0].add(jnp.einsum("b...ij,b...j->b...i", abar[0], h0.reshape(drive.shape[1:])))
    return _scan(abar, drive).reshape(shape)


def _fold_scan(abar, drive, h0):
    shape, drive = drive.shape, drive.reshape(abar.shape[:-1])
    if h0 is not None:  # fold A_1 h0 into the first drive
        drive = drive.at[0].add(jnp.einsum("b...ij,b...j->b...i", abar[0], h0.reshape(drive.shape[1:])))
    return _scan(abar, drive).reshape(shape)


def part_dense(p, leaf, w, v, low, noise, h0=None):
    """Branch with leaf weights spread over all leaves, W (T, B, leaves), so every leaf read is one dense matmul
    instead of a per-token gather. v (T, B, rank) -> states (T, B, n), output P_r h (T, B, out_rank)."""
    n_leaves, n = p["c"].shape
    W = jnp.einsum("tbq,tbql->tbl", w, jax.nn.one_hot(leaf, n_leaves, dtype=w.dtype))
    outer = lambda a: (W[..., None] * a[..., None, :]).reshape(W.shape[:2] + (-1,))
    flat = lambda u: jnp.swapaxes(u, 1, 2).reshape(-1, n)  # (leaves, n, r) -> (leaves r, n)
    drive = outer(v) @ flat(p["u_in"]) + W @ p["c"]
    if "v_low" in p:
        drive = drive + outer(low @ p["v_low"].T) @ flat(p["u_low"])
    if noise is not None:
        drive = drive + noise
    a = transitions(p)
    h = _fold_scan((W @ a.reshape(n_leaves, -1)).reshape(W.shape[:2] + a.shape[1:]), drive, h0)
    y = (h @ jnp.moveaxis(p["p_out"], 2, 0).reshape(n, -1)).reshape(W.shape[:2] + p["p_out"].shape[:2])
    if p["p_out"].shape[0] == 1:
        return h, y[:, :, 0]
    return h, jnp.einsum("tbl,tblr->tbr", W, y)


def _mm(a, b):
    """Large-vocabulary form: a @ b with LM_DTYPE operands and fp32 accumulation and result."""
    return jnp.matmul(a.astype(LM_DTYPE), b.astype(LM_DTYPE), preferred_element_type=jnp.float32)


def _grouped_mm(x, gid, wts, blk=64):
    """x (R, k), group ids gid (R,), wts (G, k, m) -> (R, m), row i = x_i wts[gid_i]. On GPU: grouped_mm.py (no padding).
    Otherwise rows are sorted by group and each group is padded to whole blocks of blk rows, so every block is one dense
    matmul with one gathered weight matrix."""
    R, G = x.shape[0], wts.shape[0]  # rows with gid >= G give zeros
    if _use_kernel(None):
        import grouped_mm
        if grouped_mm.fits(R, x.shape[1], wts.shape[2]):
            return grouped_mm.grouped_mm(x, gid, wts, LM_DTYPE)
    wts = jnp.concatenate([wts, jnp.zeros_like(wts[:1])])
    gid, G = jnp.minimum(gid, G), G + 1
    order = jnp.argsort(gid)
    gs = gid[order]
    count = jnp.bincount(gid, length=G)
    padded = -(-count // blk) * blk
    end = jnp.cumsum(padded)
    pos = (end - padded)[gs] + jnp.arange(R) - (jnp.cumsum(count) - count)[gs]
    nb = R // blk + G  # bound on the number of blocks
    buf = jnp.zeros((nb * blk, x.shape[1]), x.dtype).at[pos].set(x[order])
    bg = jnp.minimum(jnp.searchsorted(end, jnp.arange(nb) * blk, side="right"), G - 1)
    y = jnp.einsum("bik,bkm->bim", buf.reshape(nb, blk, -1).astype(LM_DTYPE), wts[bg].astype(LM_DTYPE),
                   preferred_element_type=jnp.float32).reshape(nb * blk, -1)[pos]
    return jnp.zeros_like(y).at[order].set(y)


def _use_kernel(T):
    """Pallas kernels (leaf_scan.py, grouped_mm.py): on GPU, with T a multiple of the scan chunk."""
    import leaf_scan
    return LM_KERNEL and (T is None or T % leaf_scan.CHUNK == 0) and jax.default_backend() == "gpu"


def _scan_table(tab, leaf, w, drive, h0):
    """States of a branch: A_t = tab[leaf slot 0], straight-through weights w; drive (T, B, n)."""
    T, B = leaf.shape[:2]
    x = drive.reshape((T, B) + tab.shape[1:-1])
    if h0 is not None:
        x = x.at[0].add(jnp.einsum("b...ij,b...j->b...i", tab[leaf[0, :, 0]], h0.reshape(x.shape[1:])))
    if _use_kernel(T):
        import leaf_scan
        import scan
        f = scan.table_scan if SCAN_IMPL == "scan" else leaf_scan.leaf_scan
        return f(tab, leaf.astype(jnp.int32), w, x).reshape(drive.shape)
    abar = jnp.einsum("tbq,tbq...->tb...", w, tab[leaf])
    return _scan(abar, x).reshape(drive.shape)


def part_grouped(p, leaf, w, v, low, noise, h0=None):
    """Branch whose K leaf reads per token are grouped matmuls over the (token, slot) rows sorted by leaf; A and c are
    read as W (T, B, leaves) times the leaf table. v (T, B, rank) -> states (T, B, n), output P_r h (T, B, out_rank)."""
    n_leaves, n = p["c"].shape
    T, B, K = leaf.shape
    W = jnp.einsum("tbq,tbql->tbl", w, jax.nn.one_hot(leaf, n_leaves, dtype=w.dtype))
    rows = lambda a: jnp.broadcast_to(a[:, :, None], (T, B, K, a.shape[-1])).reshape(T * B * K, -1)
    mix = lambda y: jnp.einsum("tbq,tbqk->tbk", w, y.reshape(T, B, K, -1))
    lv = _mm(low, p["v_low"].T) if "v_low" in p else None
    if leaves.ON and _use_kernel(None) and leaves.fits(v.shape[-1], n, p["p_out"].shape[1]) and \
            (lv is None or leaves.fits(lv.shape[-1], n)):  # leaves.py; U v + U_low (V_low nu) in one kernel, no concat
        pln, wf = leaves.plan(leaf.reshape(T * B, K), n_leaves), w.reshape(T * B, K)
        xs, us = ((v,), (p["u_in"],)) if lv is None else ((v, lv), (p["u_in"], p["u_low"]))
        drive = leaves.routed_mm(tuple(a.reshape(T * B, -1) for a in xs), us, wf, pln, p["c"], LM_DTYPE)
        drive = drive.reshape(T, B, n)
        if noise is not None:
            drive = drive + noise
        h = _scan_table(leaves.leaf_blocks(p["skew"], p["d_raw"]), leaf, w, drive, h0)
        if p["p_out"].shape[0] == 1:
            return h, _mm(h, p["p_out"][0].T)
        return h, leaves.routed_mm(h.reshape(T * B, n), p["p_out"], wf, pln, None, LM_DTYPE).reshape(T, B, -1)
    x, u = (v, p["u_in"]) if lv is None else (jnp.concatenate([v, lv], -1), jnp.concatenate([p["u_in"], p["u_low"]], -1))
    gid = leaf.reshape(-1)
    drive = mix(_grouped_mm(rows(x), gid, jnp.swapaxes(u, 1, 2))) + _mm(W, p["c"])
    if noise is not None:
        drive = drive + noise
    h = _scan_table(transitions(p), leaf, w, drive, h0)
    if p["p_out"].shape[0] == 1:
        return h, _mm(h, p["p_out"][0].T)
    return h, mix(_grouped_mm(rows(h), gid, jnp.swapaxes(p["p_out"], 1, 2)))


def part_parallel(p, leaf, w, v, low, noise, h0=None):
    """part_grouped with the leaves sharded by leaf id over MESH and the tokens sharded by batch row. Every device
    gathers the tokens' rows, runs its own leaves, and the partial drives and outputs are reduce-scattered back to the
    tokens' devices. The A and c tables are small and gathered whole."""
    from jax.sharding import PartitionSpec as P
    ax, keys = "x", [k for k in LEAF_KEYS if k in p]
    assert p["p_out"].shape[0] > 1 and noise is None

    def f(big, v_low, leaf, w, v, low, h0):
        nl, n = big["c"].shape
        K = leaf.shape[-1]
        gat = lambda a: jax.lax.all_gather(a, ax, axis=1, tiled=True)
        scat = lambda a: jax.lax.psum_scatter(a, ax, scatter_dimension=1, tiled=True)
        tab = jax.lax.all_gather(transitions(big), ax, tiled=True)
        c = jax.lax.all_gather(big["c"], ax, tiled=True)
        lg, wg = gat(leaf), gat(w)
        gid = (lg - jax.lax.axis_index(ax) * nl).reshape(-1)
        gid = jnp.where((gid >= 0) & (gid < nl), gid, nl)
        rows = lambda a: jnp.broadcast_to(a[:, :, None], a.shape[:2] + (K, a.shape[-1])).reshape(-1, a.shape[-1])
        mix = lambda y: jnp.einsum("tbq,tbqk->tbk", wg, y.reshape(wg.shape + (-1,)))
        x, u = v, big["u_in"]
        if v_low is not None:
            x, u = jnp.concatenate([v, _mm(low, v_low.T)], -1), jnp.concatenate([u, big["u_low"]], -1)
        W = jnp.einsum("tbq,tbql->tbl", w, jax.nn.one_hot(leaf, c.shape[0], dtype=w.dtype))
        drive = scat(mix(_grouped_mm(rows(gat(x)), gid, jnp.swapaxes(u, 1, 2)))) + _mm(W, c)
        h = _scan_table(tab, leaf, w, drive, h0)
        y = scat(mix(_grouped_mm(rows(gat(h)), gid, jnp.swapaxes(big["p_out"], 1, 2))))
        return h, y

    tok = P(None, ax)
    args = ({k: p[k] for k in keys}, p.get("v_low"), leaf, w, v, low, h0)
    specs = ({k: P(ax) for k in keys}, None if p.get("v_low") is None else P(), tok, tok, tok, tok,
             None if h0 is None else P(ax))
    return jax.shard_map(f, mesh=MESH, in_specs=specs, out_specs=(tok, tok), check_vma=False)(*args)


def leaf_sharding(params):
    """NamedShardings for params over MESH: branch leaf arrays and tree levels with at least one node per device are
    split along the leaf axis, everything else is replicated."""
    from jax.sharding import NamedSharding, PartitionSpec as P
    nd = MESH.devices.size

    def spec(path, x):
        keys = [str(getattr(k, "key", "")) for k in path]
        split = "branches" in keys and any(k in LEAF_KEYS for k in keys) and x.shape[0] % nd == 0
        return NamedSharding(MESH, P("x") if split and x.shape[0] >= nd else P())
    return jax.tree_util.tree_map_with_path(spec, params)


def root_dense(p, z, noise, h0=None, ids=None, max_distinct=None):
    """Root with a rank-r leaf table: leaf x = base + z(x) G, z (T, B, r). -> states, output P h."""
    if leaves.ON and ids is not None and _use_kernel(z.shape[0]):  # leaves.py: blocks once per distinct token id
        tab, idx = leaves.token_blocks(p["skew"][0], p["d_raw"][0], p["g_s"], p["g_d"], z, ids, LM_DTYPE,
                                       max_distinct=max_distinct)
        drive = p["c"][0] + _mm(z, p["g_c"])
        if noise is not None:
            drive = drive + noise
        h = _scan_table(tab, idx[..., None], jnp.ones(idx.shape + (1,), drive.dtype), drive, h0)
        return h, _mm(h, p["p_out"][0].T)
    sk = p["skew"][0] + _mm(z, p["g_s"]).reshape(z.shape[:2] + p["skew"].shape[1:])
    sk = 0.5 * (sk - jnp.swapaxes(sk, -1, -2))
    d = p["d_raw"][0] + _mm(z, p["g_d"])
    a = jnp.clip(d, -1.0, 1.0).reshape(sk.shape[:-1])[..., None] * jax.checkpoint(_expm_skew)(sk)
    drive = p["c"][0] + _mm(z, p["g_c"])
    if noise is not None:
        drive = drive + noise
    if _use_kernel(z.shape[0]):
        import leaf_scan
        x = drive.reshape(a.shape[:-1])
        if h0 is not None:
            x = x.at[0].add(jnp.einsum("b...ij,b...j->b...i", a[0], h0.reshape(x.shape[1:])))
        h = leaf_scan.token_scan(a, x).reshape(drive.shape)
    else:
        h = _fold_scan(a, drive, h0)
    return h, _mm(h, p["p_out"][0].T)


def forward(params, tokens, beam=8, tau=ROUTE_TAU, mode="hard", key=None, route_noise=0.0, noise_std=0.0, state=None,
            return_state=False, prev=None, remat=False, targets=None, max_distinct=None):
    """tokens (B, T) -> log-probs (B, T, classes). mode: "hard" or "st" (straight-through beam gradient).
    targets (B, T), with --emb only: return -log p(targets) (B, T) from a chunked cross-entropy instead.
    max_distinct, with --emb: rows of the root table (distinct token ids it holds); None: root_cap(tokens, vocab).
    A batch with more distinct ids than this is wrong for the extra ids; train() checks every batch."""
    xs = jnp.swapaxes(tokens, 0, 1)
    parts = [params["root"]] + [leaf_table(p) for p in params["branches"]]
    low, logits, last = None, params["head_b"], []
    if "emb" in params:
        return _forward_emb(params, xs, parts, beam, tau, mode, key, route_noise, noise_std, state, return_state, remat,
                            targets, max_distinct)
    for i, p in enumerate(parts):
        pkey = None if key is None else jax.random.fold_in(key, i)
        noise = None
        if pkey is not None and noise_std > 0:
            noise = noise_std * jax.random.normal(jax.random.fold_in(pkey, 2), xs.shape + p["c"].shape[-1:])
        if i == 0:  # root leaf = token
            leaf = xs if "context" not in p else \
                context_leaf(xs, p["context"].shape[0], p["v_tok"].shape[0], p["c"].shape[0], prev)
            leaf, w = leaf[:, :, None], jnp.ones(xs.shape + (1,), jnp.float32)
        else:
            f = p["tok"][xs] if p["w_node"].shape[1] == p["tok"].shape[1] else jnp.concatenate([p["tok"][xs], low], -1)
            leaf, w = walk(p, f, beam, tau, mode, None if pkey is None else jax.random.fold_in(pkey, 1), route_noise)
        h = part(p, leaf, w, xs, low, noise, None if state is None else state[i])
        last.append(h[-1])
        low = _rms(h)
        out = leaf if p["p_out"].shape[0] > 1 else 0 * leaf
        logits = logits + jnp.einsum("tbq,tbqrk,tbk->tbr", w, p["p_out"][out], h) @ p["q_out"].T
    logp = jnp.swapaxes(jax.nn.log_softmax(logits, -1), 0, 1)
    return (logp, last) if return_state else logp


def root_cap(tokens, vocab):
    """Default root-table rows for a micro-batch of `tokens` positions (leaves.distinct_cap)."""
    return leaves.distinct_cap(tokens, vocab)


def _forward_emb(params, xs, parts, beam, tau, mode, key, route_noise, noise_std, state, return_state, remat,
                 targets=None, max_distinct=None):
    """forward with the shared, tied embedding (init emb > 0)."""
    e = _rms(params["emb"][xs])
    part_lm = part_parallel if MESH is not None else part_grouped if LEAF_MM == "grouped" else part_dense
    root_f, part_f = (jax.checkpoint(root_dense, static_argnums=(5,)), jax.checkpoint(part_lm)) if remat else \
        (root_dense, part_lm)
    md = max_distinct or root_cap(xs.size, params["emb"].shape[0])
    low, o, last = None, 0.0, []
    for i, p in enumerate(parts):
        pkey = None if key is None else jax.random.fold_in(key, i)
        noise = None
        if pkey is not None and noise_std > 0:
            noise = noise_std * jax.random.normal(jax.random.fold_in(pkey, 2), xs.shape + p["c"].shape[-1:])
        h0 = None if state is None else state[i]
        if i == 0:
            h, y = root_f(p, _mm(e, p["r_in"]), noise, h0, xs, md)
        else:
            f = e if p["w_node"].shape[1] == e.shape[-1] else jnp.concatenate([e, low], -1)
            leaf, w = walk(p, f, beam, tau, mode, None if pkey is None else jax.random.fold_in(pkey, 1), route_noise,
                           dense=True)
            h, y = part_f(p, leaf, w, _mm(e, p["w_v"]), low, noise, h0)
        last.append(h[-1])
        low = _rms(h)
        o = o + _mm(y, p["q_out"].T)
    hf = _rms(o) * params["out_g"]
    if targets is not None:
        import head  # fused chunked cross-entropy: no (tokens, vocab) logits stored
        impl = "pallas" if _use_kernel(None) and LM_DTYPE == jnp.bfloat16 else "chunk"
        nll = head.fused_ce(hf, params["emb"], params["head_b"], jnp.swapaxes(targets, 0, 1), dtype=LM_DTYPE,
                            return_per_token=True, impl=impl)
        nll = jnp.swapaxes(nll, 0, 1)
        return (nll, last) if return_state else nll
    logits = _mm(hf, params["emb"].T) + params["head_b"]
    logp = jnp.swapaxes(jax.nn.log_softmax(logits, -1), 0, 1)
    return (logp, last) if return_state else logp


# ----------------------------------------------------------------------------------------------------------- tasks

def _perm_table(perms):
    index = {q: i for i, q in enumerate(perms)}
    return np.array([[index[tuple(b[a[j]] for j in range(5))] for b in perms] for a in perms], np.int32)


def _even(q):
    q, s = list(q), 0
    for i in range(len(q)):
        while q[i] != i:
            j = q[i]
            q[i], q[j] = q[j], q[i]
            s ^= 1
    return s == 0


class Group:
    """Word problem: label at t = x_1 ... x_t."""

    def __init__(self, table):
        self.table, self.vocab = table, len(table)
        self.classes, self.identity = len(table), 0

    def sample(self, rng, batch, length):
        x = rng.integers(0, self.vocab, (batch, length), dtype=np.int32)
        y, s = np.empty_like(x), np.full(batch, self.identity, np.int32)
        for t in range(length):
            s = self.table[s, x[:, t]]
            y[:, t] = s
        return x, y, np.ones_like(x, bool)


class Parity:
    vocab, classes = 2, 2

    def sample(self, rng, batch, length):
        x = rng.integers(0, 2, (batch, length), dtype=np.int32)
        return x, np.cumsum(x, 1) % 2, np.ones_like(x, bool)


class Cycle:
    vocab, classes = 3, 5

    def sample(self, rng, batch, length):
        x = rng.integers(0, 3, (batch, length), dtype=np.int32)
        return x, np.cumsum(x - 1, 1) % 5, np.ones_like(x, bool)


class ModArith:
    """Residues 0..4 alternating with + - * (tokens 5..7), mod 5, labelled after each residue."""
    vocab, classes = 8, 5

    def sample(self, rng, batch, length):
        x = np.empty((batch, length), np.int32)
        x[:, ::2] = rng.integers(0, 5, (batch, (length + 1) // 2))
        x[:, 1::2] = rng.integers(5, 8, (batch, length // 2))
        y = np.zeros((batch, length), np.int32)
        acc, term = np.zeros(batch, np.int64), x[:, 0].astype(np.int64)
        y[:, 0] = term % 5
        for t in range(2, length, 2):
            op, r = x[:, t - 1], x[:, t].astype(np.int64)
            acc = np.where(op != 7, acc + term, acc)
            term = np.where(op == 5, r, np.where(op == 6, -r, term * r)) % 5
            y[:, t] = (acc + term) % 5
        valid = np.zeros((batch, length), bool)
        valid[:, ::2] = True
        return x, y, valid


_PERMS = list(itertools.permutations(range(5)))
STATE_TASKS = {  # name: (generator, train length, test length)
    "s5_word": (lambda: Group(_perm_table(_PERMS)), 32, 500),
    "a5_word": (lambda: Group(_perm_table([q for q in _PERMS if _even(q)])), 32, 500),
    "parity_check": (Parity, 40, 256),
    "cycle_navigation": (Cycle, 40, 256),
    "modular_arithmetic": (ModArith, 40, 256),
}
LM_TASKS = {"enwik8": 256, "hg38": 5, "fineweb_edu": 50257, "fineweb_edu_1b": 50257,
            "fineweb_edu_3b": 50257}  # name: vocabulary (hg38: A C G T = 0..3, N = 4; fineweb_edu: GPT-2)
CLF_TASKS = {"dummy_mouse_enhancers_ensembl": 2, "demo_coding_vs_intergenomic_seqs": 2, "demo_human_or_worm": 2,
             "human_enhancers_cohn": 2, "human_enhancers_ensembl": 2, "human_ensembl_regulatory": 3,
             "human_nontata_promoters": 2, "human_ocr_ensembl": 2}  # Genomic Benchmarks: name: classes
SEQ_LENGTHS = (32, 64, 128, 256, 500)
N_EVAL, EVAL_CHUNK = 512, 128


class StateTask:

    def __init__(self, name, seed):
        make, self.train_len, self.eval_len = STATE_TASKS[name]
        self.gen = make()
        self.vocab, self.classes = self.gen.vocab, self.gen.classes
        self.sets = {s: self.gen.sample(np.random.default_rng(o + seed), N_EVAL, self.eval_len)
                     for s, o in (("val", 20_000), ("test", 10_000))}

    def batch(self, rng, size):
        x, y, valid = self.gen.sample(rng, size, self.train_len)
        return x, y, valid.astype(np.float32)

    def carry_batch(self, rng, size, frac):
        """(x, y, mask, prefix, use); rows with use = 1 continue the stream that `prefix` starts."""
        L = self.train_len
        use = (rng.random(size) < frac).astype(np.float32)
        x, y, valid = self.gen.sample(rng, size, L)
        sx, sy, sv = self.gen.sample(rng, size, 2 * L)
        prefix = sx[:, :L]
        pick = use[:, None] > 0
        x, y, valid = np.where(pick, sx[:, L:], x), np.where(pick, sy[:, L:], y), np.where(pick, sv[:, L:], valid)
        return x, y, valid.astype(np.float32), prefix, use

    def evaluate(self, predict, params, split):
        """Sequence accuracy per length."""
        x, y, valid = self.sets[split]
        pred = np.concatenate([np.asarray(predict(params, x[i:i + EVAL_CHUNK])) for i in range(0, len(x), EVAL_CHUNK)])
        ok = np.cumprod(np.where(valid, pred == y, True), axis=1)
        return {str(L): float(ok[:, L - 1].mean()) for L in SEQ_LENGTHS if L <= self.eval_len}


class LMTask:
    """Next-token bits per token. enwik8: 90M / 5M / 5M split; hg38: HyenaDNA split.
    fineweb_edu: nats per token; val.bin (10M held-out tokens) gives val from its first half, test from its second."""
    higher = False

    def __init__(self, name, cfg):
        self.metric, self.unit, self.chunk = "val_bpc", LN2, 64
        if name == "enwik8":
            raw = np.frombuffer((ROOT / "data" / name).read_bytes(), np.uint8)
            self.train, val, test = raw[:90_000_000], raw[90_000_000:95_000_000], raw[95_000_000:]
        elif name.startswith("fineweb_edu"):  # fineweb_edu_1b, _3b: same val.bin, 1.05B or 2.99B training tokens
            d = ROOT / "data" / name
            self.train, held = (np.memmap(d / f"{s}.bin", np.uint16, "r") for s in ("train", "val"))
            val, test = held[:len(held) // 2], held[len(held) // 2:]
            self.metric, self.unit, self.chunk = "val_nats", 1.0, 16
        else:
            self.train, val, test = (np.memmap(ROOT / "data" / name / f"{s}.u8", np.uint8, "r")
                                     for s in ("train", "valid", "test"))
        self.vocab = self.classes = LM_TASKS[name]
        self.T = cfg.T
        self.streams = {"val": val, "test": test}
        self.sets = {"val": self._windows(val, cfg.val_windows), "test": self._windows(test, cfg.test_chars // cfg.T)}

    def _windows(self, ids, m):
        m = min(m, (len(ids) - 1) // self.T)
        ids = np.asarray(ids[:m * self.T + 1], np.int32)
        return ids[:-1].reshape(m, self.T), ids[1:].reshape(m, self.T)

    def batch(self, rng, size):
        idx = rng.integers(0, len(self.train) - self.T - 1, size)[:, None] + np.arange(self.T)[None]
        return self.train[idx].astype(np.int32), self.train[idx + 1].astype(np.int32), np.ones((size, self.T), np.float32)

    def carry_batch(self, rng, size, frac):
        """(x, y, mask, prefix, use); prefix is the preceding window."""
        use = (rng.random(size) < frac).astype(np.float32)
        idx = rng.integers(self.T, len(self.train) - self.T - 1, size)[:, None] + np.arange(self.T)[None]
        x, y, prefix = (self.train[idx + k].astype(np.int32) for k in (0, 1, -self.T))
        return x, y, np.ones((size, self.T), np.float32), prefix, use

    def evaluate(self, nll_fn, params, split):
        x, y, c = *self.sets[split], self.chunk
        nll = sum(float(nll_fn(params, x[i:i + c], y[i:i + c])) for i in range(0, len(x), c))
        return nll / y.size / self.unit

    def evaluate_stream(self, params, split, chunk=16384, limit=0):
        """Bits per token over the split as one stream, states carried across chunks."""
        ids = self.streams[split]
        if limit > 0:
            ids = ids[:limit + 1]
        n = len(ids) - 1
        params = fold_tree(params)
        st = [jnp.zeros((1, p["c"].shape[-1]), jnp.float32) for p in [params["root"]] + list(params["branches"])]
        nll = 0.0
        for i in range(0, n, chunk):
            j = min(n, i + chunk)
            x = np.asarray(ids[i:j], np.int32)[None]
            y = np.asarray(ids[i + 1:j + 1], np.int32)[None]
            prev = None
            if "context" in params["root"]:
                k = params["root"]["context"].shape[0]
                prev = np.concatenate([np.zeros(k - 1, np.int32), np.asarray(ids[max(0, i - k + 1):i], np.int32)])
                prev = prev[-(k - 1):][None]
            s, st = _stream_nll(params, x, y, st, prev)
            nll += float(s)
        return nll / n / LN2, n


@jax.jit
def _stream_nll(params, x, y, state, prev=None):
    """x, y (1, C) -> (-sum log p(y), states after the chunk)."""
    logp, last = forward(params, x, mode="hard", state=state, return_state=True, prev=prev)
    return -jnp.take_along_axis(logp, y[..., None], -1).sum(), last


class ClfTask:
    """Genomic Benchmarks: label read at the last base. Tokens A C G T N = 0..4, pad = 5."""
    higher, vocab = True, 6

    def __init__(self, name, seed):
        self.classes = CLF_TASKS[name]
        d = ROOT / "data" / "genomic_benchmarks" / name
        tr, te = (np.load(d / f"{s}.npz") for s in ("train", "test"))
        perm = np.random.default_rng(seed).permutation(len(tr["y"]))
        cut = len(perm) // 10
        pick = lambda z, i: (z["x"][i].astype(np.int32), z["lengths"][i], z["y"][i].astype(np.int32))
        self.sets = {"val": pick(tr, perm[:cut]), "test": pick(te, np.arange(len(te["y"])))}
        self.train = pick(tr, perm[cut:])

    @staticmethod
    def _crop(x, lengths):  # round up to 64 to limit recompiles
        return x[:, :min(x.shape[1], -(-int(lengths.max()) // 64) * 64)]

    def batch(self, rng, size):
        x, lengths, y = self.train
        i = rng.integers(0, len(y), size)
        xb = self._crop(x[i], lengths[i])
        m = np.zeros(xb.shape, np.float32)
        m[np.arange(size), lengths[i] - 1] = 1.0
        return xb, np.broadcast_to(y[i][:, None], xb.shape).copy(), m

    def evaluate(self, predict, params, split):
        x, lengths, y = self.sets[split]
        order, hits = np.argsort(lengths), 0
        for j in range(0, len(order), EVAL_CHUNK):
            i = order[j:j + EVAL_CHUNK]
            pred = np.asarray(predict(params, self._crop(x[i], lengths[i])))
            hits += int((pred[np.arange(len(i)), lengths[i] - 1] == y[i]).sum())
        return hits / len(y)


# -------------------------------------------------------------------------------------------------------- training

LEAF_KEYS = ("u_in", "c", "p_out", "skew", "d_raw", "u_low")
DECAYED = ("emb", "u_in", "u_low", "p_out", "q_out", "w_v", "v_low", "w_node", "r_in", "g_c")  # weight decay:
# dense maps and the embedding; not the transitions (skew, d_raw and their generators), biases or the norm gain


def to_tree(p, depth):
    """Leaf j = (D + 1)^-1/2 * sum of the increments on its root-to-leaf path; deeper levels start at zero."""
    sc = (depth + 1) ** -0.5
    levels = {k: [p[k][:1] / sc] + [jnp.zeros((2**d,) + p[k].shape[1:], p[k].dtype) for d in range(1, depth + 1)]
              for k in LEAF_KEYS if k in p}
    return dict({k: v for k, v in p.items() if k not in levels}, levels=levels)


def leaf_table(p):
    """Tree of increments -> per-leaf arrays."""
    if "levels" not in p:
        return p
    out = {k: v for k, v in p.items() if k != "levels"}
    for k, levels in p["levels"].items():
        depth = len(levels) - 1
        if leaves.ON:  # same sum, one tree level at a time in the backward
            out[k] = leaves.tree_sum(list(levels))
            continue
        out[k] = (depth + 1) ** -0.5 * sum(jnp.repeat(lv, 2**(depth - d), axis=0) for d, lv in enumerate(levels))
    return out


def fold_tree(params):
    return dict(params, branches=[leaf_table(p) for p in params["branches"]])


def init_lm(cfg, vocab):
    """Large-vocabulary params as trees of increments, built in one jit (sharded over MESH when set) without drawing
    the leaves that to_tree discards. -> params, stored parameter count of the folded leaf tables."""
    def make(key):
        p = init(key, vocab, vocab, cfg.n, cfg.block, cfg.depth, cfg.layers, cfg.feat, cfg.rank, cfg.out_rank,
                 cfg.lower_rank, cfg.shared_out, "dyn", 1, 16, cfg.emb, cfg.root_rank, branch_leaves=1)
        return dict(p, branches=[to_tree(q, cfg.depth) for q in p["branches"]])
    key = jax.random.PRNGKey(cfg.seed)
    shapes = jax.eval_shape(make, key)
    count = 0
    for q in [shapes["root"], shapes["emb"], shapes["head_b"], shapes["out_g"]] + shapes["branches"]:
        q = dict(q) if isinstance(q, dict) else {"x": q}
        levels = q.pop("levels", {})
        count += sum(int(np.prod(v.shape)) for v in jax.tree_util.tree_leaves(q))
        count += sum(2**cfg.depth * int(np.prod(lv[0].shape[1:])) for lv in levels.values())
    out = None if MESH is None else leaf_sharding(shapes)
    return jax.jit(make, out_shardings=out)(key), count


def save(path, params):
    flat = {"/".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in kp): np.asarray(v)
            for kp, v in jax.tree_util.tree_flatten_with_path(params)[0]}
    np.savez(path, **flat)


def train(cfg):
    global LM_DTYPE, LM_KERNEL, MESH
    LM_DTYPE, LM_KERNEL = getattr(jnp, cfg.dtype), cfg.kernels
    if cfg.leaf_shards > 1:
        MESH = jax.sharding.Mesh(np.array(jax.devices()[:cfg.leaf_shards]), ("x",))
    lm = cfg.task in LM_TASKS
    task = LMTask(cfg.task, cfg) if lm else ClfTask(cfg.task, cfg.seed) if cfg.task in CLF_TASKS else \
        StateTask(cfg.task, cfg.seed)
    tok_sharding = None
    if cfg.emb:
        params, n_params = init_lm(cfg, task.vocab)
        if MESH is not None:
            tok_sharding = jax.sharding.NamedSharding(MESH, jax.sharding.PartitionSpec("x"))
    else:
        params = init(jax.random.PRNGKey(cfg.seed), task.vocab, task.classes, cfg.n, cfg.block, cfg.depth, cfg.layers,
                      cfg.feat, cfg.rank, cfg.out_rank, cfg.lower_rank, cfg.shared_out, cfg.branch, cfg.context_len,
                      cfg.context_bits)
        n_params = int(sum(np.prod(v.shape) for v in jax.tree_util.tree_leaves(params)))
        params["branches"] = [to_tree(p, cfg.depth) for p in params["branches"]]
    warmup = min(1000, cfg.steps // 10) if cfg.warmup_frac is None else int(round(cfg.warmup_frac * cfg.steps))
    sched = optax.warmup_cosine_decay_schedule(0.0, cfg.lr, warmup, cfg.steps, cfg.lr * 0.01)
    fast = cfg.emb and cfg.opt == "fast"
    if fast:  # large-vocabulary form: one pass per array, m and v in bf16 (sparse_opt.py)
        import sparse_opt
        opt = sparse_opt.FastAdam(sched, b2=cfg.beta2, clip=1.0, mu_dtype=jnp.bfloat16, nu_dtype=jnp.bfloat16,
                                  branch_lr=cfg.branch_lr, weight_decay=cfg.weight_decay,
                                  decay=lambda keys: any(k in DECAYED for k in keys))
    elif cfg.weight_decay or cfg.beta2 != 0.999:  # AdamW as the baselines run it: decay on arrays of rank >= 2
        decay = lambda p: jax.tree_util.tree_map(lambda x: x.ndim >= 2, p)
        opt = optax.apply_if_finite(optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(
            sched, b2=cfg.beta2, weight_decay=cfg.weight_decay, mask=decay)), 1000)
    else:
        opt = optax.apply_if_finite(optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched)), 1000)
    opt_state = opt.init(params)

    def loss_fn(p, x, y, m, key, prefix=None, use=None, md=None):
        state = None
        if prefix is not None:
            last = forward(p, prefix, cfg.beam, ROUTE_TAU, "st", jax.random.fold_in(key, 7), cfg.route_noise,
                           cfg.state_noise, return_state=True)[1]
            state = [jax.lax.stop_gradient(h) * use[:, None] for h in last]
        if cfg.emb:
            nll = forward(p, x, cfg.beam, ROUTE_TAU, "st", key, cfg.route_noise, cfg.state_noise, state=state,
                          remat=cfg.remat, targets=y, max_distinct=md)
            return (nll * m / m.sum()).sum()
        logp = forward(p, x, cfg.beam, ROUTE_TAU, "st", key, cfg.route_noise, cfg.state_noise, state=state,
                       remat=cfg.remat)
        return -(jnp.take_along_axis(logp, y[..., None], -1)[..., 0] * m / m.sum()).sum()

    def grad_fn(p, x, y, m, key, prefix=None, use=None, md=None):
        if fast and MESH is None:  # gradient of the folded leaf tables; FastAdam forms the level gradients itself
            p = fold_tree(p)
        if cfg.accum == 1:
            return jax.value_and_grad(loss_fn)(p, x, y, m, key, prefix, use, md)
        assert prefix is None, "--accum without --carry"
        split = lambda a: a.reshape((cfg.accum, -1) + a.shape[1:])

        def body(acc, mb):  # equal-size micro-batches, mean of their losses
            ce, g = jax.value_and_grad(loss_fn)(p, *mb[:3], jax.random.fold_in(key, mb[3]), md=md)
            return jax.tree_util.tree_map(jnp.add, acc, (ce, g)), None

        zero = (jnp.zeros(()), jax.tree_util.tree_map(jnp.zeros_like, p))
        (ce, g), _ = jax.lax.scan(body, zero, (split(x), split(y), split(m), jnp.arange(cfg.accum)))
        return jax.tree_util.tree_map(lambda v: v / cfg.accum, (ce, g))

    def step(p, s, x, y, m, key, prefix=None, use=None, md=None):
        ce, g = grad_fn(p, x, y, m, key, prefix, use, md)
        if fast:
            p, s, _ = opt.step(p, g, s)
            return p, s, ce
        u, s = opt.update(g, s, p)
        if cfg.branch_lr != 1.0:
            u = dict(u, branches=jax.tree_util.tree_map(lambda v: v * cfg.branch_lr, u["branches"]))
        return optax.apply_updates(p, u), s, ce

    step = jax.jit(step, donate_argnums=(0, 1) if fast else (), static_argnames=("md",))  # update in place
    micro = cfg.batch // cfg.accum * cfg.T
    cap = root_cap(micro, task.vocab) if cfg.emb else None
    distinct = {"max": 0, "over": 0}  # root table: most distinct ids in a micro-batch, steps run with the full table
    predict = jax.jit(lambda p, x: forward(p, x, mode="hard").argmax(-1))
    nll = jax.jit(lambda p, x, y: forward(p, x, mode="hard", targets=y, max_distinct=min(x.size, task.vocab)).sum()
                  if cfg.emb else
                  -jnp.take_along_axis(forward(p, x, mode="hard"), y[..., None], -1).sum())

    name = f"{cfg.task}_D{cfg.depth}_n{cfg.n}_b{cfg.block}_K{cfg.beam}" + ("" if cfg.branch == "dyn" else f"_{cfg.branch}") \
        + ("" if cfg.context_len == 1 else f"_k{cfg.context_len}") + f"{cfg.tag}_s{cfg.seed}"
    out = Path(cfg.out) / name
    out.parent.mkdir(parents=True, exist_ok=True)
    wb = None
    if cfg.wandb:
        import wandb
        wb = wandb.init(name=os.environ.get("WANDB_NAME", name), config=dict(vars(cfg), n_params=n_params))
    print(name, n_params, jax.devices()[0], flush=True)
    rng, log, t0 = np.random.default_rng(cfg.seed), [], time.time()
    best = {"val": float("inf") if lm else -1.0}
    for it in range(1, cfg.steps + 1):
        prefix = use = None
        if cfg.carry > 0:
            assert isinstance(task, (StateTask, LMTask)), "--carry: state and LM tasks"
            x, y, m, prefix, use = task.carry_batch(rng, cfg.batch, cfg.carry)
        else:
            x, y, m = task.batch(rng, cfg.batch)
            if tok_sharding is not None:
                x, y, m = jax.device_put((x, y, m), tok_sharding)
        md = None
        if cfg.emb:  # a micro-batch with more distinct ids than the root table holds runs with the full table
            most = max(len(np.unique(c)) for c in np.asarray(x).reshape(cfg.accum, -1))
            distinct["max"] = max(distinct["max"], most)
            if most > cap:
                md, distinct["over"] = min(micro, task.vocab), distinct["over"] + 1
        params, opt_state, ce = step(params, opt_state, x, y, m, jax.random.PRNGKey(cfg.seed * 1_000_003 + it),
                                     prefix, use, md=md)
        if it % cfg.eval_every and it != cfg.steps:
            continue
        rec = {"step": it, "loss": float(ce), "time_s": round(time.time() - t0, 1)}
        if cfg.emb:
            rec.update(root_rows=cap, distinct_max=distinct["max"], full_table_steps=distinct["over"])
        if lm:
            rec[task.metric] = val = task.evaluate(nll, params, "val")
            if task.unit == 1.0:
                rec["val_ppl"] = float(np.exp(val))
            rec["tokens"] = it * cfg.batch * cfg.T
            better = val < best["val"]
            if cfg.stream_val > 0:
                rec["val_stream"] = task.evaluate_stream(params, "val", cfg.stream_chunk, cfg.stream_val)[0]
        elif cfg.task in CLF_TASKS:
            rec["val_acc"] = val = task.evaluate(predict, params, "val")
            better = val > best["val"]
        else:
            rec["val_seq"] = task.evaluate(predict, params, "val")
            val = rec["val_seq"][str(task.eval_len)]
            better = val > best["val"]
        if better:
            best = {"val": val, "step": it, "params": jax.device_get(params)}
            if "val_stream" in rec:
                best["val_stream"] = rec["val_stream"]
        log.append(rec)
        print(json.dumps(rec), flush=True)
        if wb is not None:
            wb.log({"loss": rec["loss"], "val": val, "best_val": best["val"],
                    **({"val_stream": rec["val_stream"]} if "val_stream" in rec else {})}, step=it)
    res = {"config": vars(cfg), "name": name, "n_params": n_params, "log": log, "best_step": best.get("step"),
           "best_val": best["val"], "train_s": round(time.time() - t0, 1),
           "code_md5": hashlib.md5(Path(__file__).read_bytes()).hexdigest()}
    if "val_stream" in best:
        res["val_stream"] = best["val_stream"]
    if "params" in best:
        del params, opt_state  # room for the best parameters on the device
        res["test"] = task.evaluate(nll if lm else predict, jax.device_put(best["params"]), "test")
        if cfg.save:
            save(out.with_suffix(".npz"), fold_tree(best["params"]))
        if lm and cfg.stream_test:
            res["test_stream"], res["test_stream_tokens"] = task.evaluate_stream(best["params"], "test",
                                                                                  cfg.stream_chunk)
    out.with_suffix(".json").write_text(json.dumps(res))
    if wb is not None:
        wb.summary.update({"best_val": best["val"], "test": res.get("test")})
        if "test_stream" in res:
            wb.summary.update({"test_stream": res["test_stream"]})
        if "val_stream" in res:
            wb.summary.update({"val_stream": res["val_stream"]})
        wb.finish()
    print("DONE", name, "best val", best["val"], "test", res.get("test"),
          *(("test_stream", res["test_stream"]) if "test_stream" in res else ()), flush=True)
    return best["val"]


def trial(cfg, argv):
    """W&B sweep trial: one process per (task, seed); objective = min over tasks of the mean best val."""
    import wandb
    wb = wandb.init()
    seeds = cfg.seeds.split(",")
    names = cfg.tasks.split(",") if cfg.tasks else [cfg.task]
    rest, skip = [], False
    for a in argv[1:]:
        if skip or a.split("=")[0] in ("--seeds", "--tasks", "--task"):
            skip = a in ("--seeds", "--tasks", "--task")
            continue
        rest.append(a)
    env = {k: v for k, v in os.environ.items() if not k.startswith("WANDB_") or k in
           ("WANDB_API_KEY", "WANDB_SILENT", "WANDB_MODE", "WANDB_ENTITY", "WANDB_DIR", "WANDB_BASE_URL")}
    env.update(WANDB_PROJECT=wb.project, WANDB_RUN_GROUP=f"trial_{wb.id}")
    out = Path(cfg.out) / wb.id
    out.mkdir(parents=True, exist_ok=True)
    procs = [subprocess.Popen([sys.executable, __file__, "train", *rest, "--task", t, "--seed", s, "--out", str(out)],
                              env=dict(env, WANDB_NAME=f"trial_{wb.id}_{t}_s{s}"),
                              stdout=open(out / f"{t}_s{s}.log", "w"), stderr=subprocess.STDOUT)
             for t in names for s in seeds]
    for q in procs:
        q.wait()
    scores, summary = [], {}
    for t in names:
        vals = []
        for s in seeds:
            hits = list(out.glob(f"{t}_*_s{s}.json"))
            vals.append(json.loads(hits[0].read_text())["best_val"] if hits else None)
            summary[f"{t}/s{s}"] = vals[-1]
        ok = [v for v in vals if v is not None]
        lm = t in LM_TASKS
        score = (-sum(ok) / len(ok) if lm else sum(ok) / len(seeds)) if ok else (-99.0 if lm else 0.0)
        summary[f"{t}/score"] = score
        scores.append(score)
    wb.summary.update({"objective": min(scores), **summary})
    wb.finish()


def sweep(path):
    """Create one W&B sweep per entry of `variants` in a YAML config."""
    import wandb
    import yaml
    cfg = yaml.safe_load(Path(path).read_text())
    param = lambda v: ({"values": v} if isinstance(v, list) else {"distribution": "log_uniform_values",
                       "min": v["log_uniform"][0], "max": v["log_uniform"][1]} if isinstance(v, dict) else {"value": v})
    paths = []
    for name, over in cfg.get("variants", {"main": {}}).items():
        values = {**cfg.get("fixed", {}), **cfg.get("search", {}), **(over or {})}
        spec = {"name": f"{Path(path).stem}-{name}", "method": cfg.get("method", "bayes"), "program": str(Path(__file__).resolve()),
                "metric": {"name": "objective", "goal": "maximize"}, "parameters": {k: param(v) for k, v in values.items()},
                "command": ["${env}", "${interpreter}", "${program}", "train", "${args}"]}
        if "run_cap" in cfg:
            spec["run_cap"] = cfg["run_cap"]
        sid = wandb.sweep(spec, project=cfg["project"])
        paths.append(f"{wandb.Api().default_entity}/{cfg['project']}/{sid}")
    print(" ".join(paths))


def parser():
    ap = argparse.ArgumentParser(description="Yggdrasil")
    ap.add_argument("command", choices=["train", "sweep"])
    ap.add_argument("config", nargs="?", help="sweep: YAML config")
    add = ap.add_argument
    add("--task", choices=[*STATE_TASKS, *LM_TASKS, *CLF_TASKS])
    add("--tasks", default="", help="comma list")
    add("--seed", type=int, default=0)
    add("--seeds", default="", help="comma list")
    add("--layers", type=int, default=2)
    add("--n", type=int, default=128)
    add("--block", type=int, default=16)
    add("--depth", type=int, default=8)
    add("--feat", type=int, default=16)
    add("--rank", type=int, default=16)
    add("--out_rank", type=int, default=8)
    add("--lower_rank", type=int, default=16)
    add("--shared_out", type=_flag, default=False)
    add("--branch", default="dyn", choices=["dyn", "out", "tok"])
    add("--context_len", type=int, default=1)
    add("--context_bits", type=int, default=16)
    add("--emb", type=int, default=0, help="shared token embedding tied to the head (large vocabularies)")
    add("--root_rank", type=int, default=0, help="rank of the root leaf table (with --emb)")
    add("--accum", type=int, default=1, help="micro-batches per step")
    add("--remat", type=_flag, default=False, help="recompute each layer in the backward pass")
    add("--dtype", default="bfloat16", choices=["bfloat16", "float32"], help="with --emb: matmul operand type")
    add("--kernels", type=_flag, default=True, help="with --emb: Pallas scan and grouped matmul kernels on GPU")
    add("--opt", default="fast", choices=["fast", "optax"], help="with --emb: sparse_opt.FastAdam (bf16 m, v) or optax")
    add("--leaf_shards", type=int, default=1, help="with --emb: devices that split the branch leaves by leaf id")
    add("--steps", type=int, default=40_000)
    add("--batch", type=int, default=256)
    add("--T", type=int, default=256)
    add("--lr", type=float, default=1e-3)
    add("--branch_lr", type=float, default=1.0)
    add("--weight_decay", type=float, default=0.0,
        help="decoupled; on the dense maps and embedding with --emb, else on arrays of rank >= 2")
    add("--beta2", type=float, default=0.999)
    add("--beam", type=int, default=8)
    add("--route_noise", type=float, default=0.05)
    add("--state_noise", type=float, default=0.0)
    add("--carry", type=float, default=0.0, help="fraction of rows with a carried state")
    add("--warmup_frac", type=float, default=None)
    add("--eval_every", type=int, default=4_000)
    add("--val_windows", type=int, default=512)
    add("--test_chars", type=int, default=5_000_000)
    add("--stream_test", type=_flag, default=False)
    add("--stream_chunk", type=int, default=16384)
    add("--stream_val", type=int, default=0)
    add("--out", default=str(ROOT / "results"))
    add("--tag", default="")
    add("--wandb", type=_flag, default=False)
    add("--save", type=_flag, default=True, help="write the best parameters (.npz)")
    return ap


def _flag(v):
    return str(v).lower() in ("1", "true", "yes")


if __name__ == "__main__":
    args = parser().parse_args()
    if args.command == "sweep":
        sweep(args.config)
    elif args.seeds:
        trial(args, sys.argv[1:])
    else:
        assert args.task, "--task (or --tasks with --seeds)"
        train(args)
