"""Fast Adam for Yggdrasil's large-vocabulary form.

FastAdam: Adam with global-norm clipping and the non-finite skip, one elementwise pass per array, m and v optionally
bf16 with stochastic rounding. The optimiser is memory bound (842M stored elements at 30M active parameters), so bf16
m and v cut its bytes per element from 32 to 24 and its state by half. Branch leaf tables stored as trees of
increments take dL/dW of the folded table and update every level in one kernel (tree_update).
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu


def _keys(path):
    return [getattr(k, "key", getattr(k, "idx", k)) for k in path]


def _get(tree, keys):
    for k in keys:
        tree = tree[k]
    return tree


def _lr(lr, count):
    return lr(count) if callable(lr) else lr


def _hash(i, seed):
    """Cheap 32-bit integer hash (no stored random numbers)."""
    h = i * jnp.uint32(0x9E3779B1) + seed
    h = (h ^ (h >> 15)) * jnp.uint32(0x85EBCA77)
    return h ^ (h >> 13)


def _store(x, dtype, seed, idx, stochastic):
    """fp32 -> dtype; bf16 with stochastic rounding when asked (unbiased: E[out] = x)."""
    if dtype != jnp.bfloat16 or not stochastic:
        return x.astype(dtype)
    bits = jax.lax.bitcast_convert_type(x, jnp.uint32) + (_hash(idx, seed) & jnp.uint32(0xFFFF))
    return jax.lax.bitcast_convert_type(bits & jnp.uint32(0xFFFF0000), jnp.float32).astype(jnp.bfloat16)


class FastAdam:
    """Adam with global-norm clipping in one elementwise pass per array (read p, g, m, v; write p, m, v), plus one
    read of the gradients for the norm.

    mu_dtype, nu_dtype: dtypes of m and v (bf16 halves their bytes). stochastic: bf16 stores use stochastic rounding.
    branch_lr: update scale for every array under "branches" (train.py --branch_lr).
    weight_decay, decay: decoupled weight decay (AdamW, p -= lr wd p) on the arrays for which decay(path) is true.
    Non-finite gradients (any NaN or inf) skip the whole step, like optax.apply_if_finite.
    Use inside a jitted step that donates params and the state, so updates are in place.
    """

    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8, clip=1.0, mu_dtype=jnp.float32, nu_dtype=jnp.float32,
                 stochastic=True, branch_lr=1.0, weight_decay=0.0, decay=None):
        self.lr, self.b1, self.b2, self.eps, self.clip = lr, b1, b2, eps, clip
        self.mu, self.nu, self.sr, self.branch_lr = mu_dtype, nu_dtype, stochastic, branch_lr
        self.wd, self.decay = weight_decay, decay

    def init(self, params):
        one = lambda x: {"m": jnp.zeros(x.shape, self.mu), "v": jnp.zeros(x.shape, self.nu)}
        return {"count": jnp.zeros((), jnp.int32), "skipped": jnp.zeros((), jnp.int32),
                "per": jax.tree_util.tree_map(one, params)}

    def _wd(self, keys):
        return self.wd if self.wd and (self.decay is None or self.decay(keys)) else 0.0

    def step(self, params, grads, state):
        """-> (params, state, info). info: global grad norm.
        grads either matches params, or holds each tree of increments {"levels": {k: [level 0..D]}} as dL/dW for the
        folded table W under k (train.py fold_tree); the level gradients are then formed inside the update."""
        flat, tdef = jax.tree_util.tree_flatten_with_path(params)
        paths, ps = [p for p, _ in flat], [x for _, x in flat]
        ss = tdef.flatten_up_to(state["per"])
        trees = {}  # (path to the branch dict, key) -> [(level, index in flat)]
        if jax.tree_util.tree_structure(grads) == tdef:
            gs = tdef.flatten_up_to(grads)
        else:
            gs = []
            for i, path in enumerate(paths):
                keys = _keys(path)
                if len(keys) >= 3 and keys[-3] == "levels":
                    trees.setdefault((tuple(keys[:-3]), keys[-2]), []).append((keys[-1], i))
                    gs.append(None)
                else:
                    gs.append(_get(grads, keys))
            fused = {}
            for (pre, k), lv in trees.items():
                G = _get(grads, list(pre) + [k])
                F = G.size // G.shape[0]
                if _tree_cols(F) is None or not _kernels():
                    for (d, i), g in zip(sorted(lv), _tree_grads(G)):  # XLA path: level gradients as arrays
                        gs[i] = g
                else:
                    fused[(pre, k)] = (G, [i for _, i in sorted(lv)])
            trees = fused
        sq = [jnp.sum(jnp.square(g.astype(jnp.float32))) for g in gs if g is not None]
        norm = jnp.sqrt(sum(sq) + sum(tree_norm(G) for G, _ in trees.values()))
        ok = jnp.isfinite(norm)
        scale = jnp.where(norm > self.clip, self.clip / norm, 1.0) if self.clip else jnp.float32(1.0)
        count = state["count"]
        lr = _lr(self.lr, count)
        new_p, new_s = list(ps), [None] * len(ps)
        t = (count + 1).astype(jnp.float32)
        for G, ids in trees.values():
            keys = _keys(paths[ids[0]])
            a = lr * (self.branch_lr if "branches" in keys else 1.0)
            pn, mn, vn = tree_update([ps[i] for i in ids], G, [ss[i]["m"] for i in ids], [ss[i]["v"] for i in ids],
                                     scale, a / (1 - self.b1**t), 1 / jnp.sqrt(1 - self.b2**t), 1 - a * self._wd(keys),
                                     ok, count, ids, self.b1, self.b2, self.eps, self.mu, self.nu, self.sr)
            for i, p2, m2, v2 in zip(ids, pn, mn, vn):
                new_p[i], new_s[i] = p2, {"m": m2, "v": v2}
        for i, (path, x, g, s) in enumerate(zip(paths, ps, gs, ss)):
            if g is None:
                continue
            keys = _keys(path)
            a = lr * (self.branch_lr if "branches" in keys else 1.0)
            seed = (count.astype(jnp.uint32) * jnp.uint32(1000003) + jnp.uint32(7919 * (i + 1)))
            p2, m2, v2 = self._dense(x, g, s, scale, a, count, seed, ok, self._wd(keys))
            new_p[i], new_s[i] = p2, {"m": m2, "v": v2}
        state = {"count": count + ok.astype(jnp.int32), "skipped": state["skipped"] + (~ok).astype(jnp.int32),
                 "per": jax.tree_util.tree_unflatten(tdef, new_s)}
        return jax.tree_util.tree_unflatten(tdef, new_p), state, {"grad_norm": norm}

    def _dense(self, x, g, s, scale, a, count, seed, ok, wd=0.0):
        b1, b2 = self.b1, self.b2
        t = (count + 1).astype(jnp.float32)
        c1, c2 = a / (1 - b1**t), 1 / jnp.sqrt(1 - b2**t)
        gg = g.astype(jnp.float32) * scale
        m = b1 * s["m"].astype(jnp.float32) + (1 - b1) * gg
        v = b2 * s["v"].astype(jnp.float32) + (1 - b2) * gg * gg
        p = x.astype(jnp.float32) * (1 - a * wd) - c1 * m / (jnp.sqrt(v) * c2 + self.eps)
        idx = jax.lax.iota(jnp.uint32, x.size).reshape(x.shape)
        mo, vo = _store(m, self.mu, seed, idx, self.sr), _store(v, self.nu, seed ^ jnp.uint32(0x5bd1e995), idx,
                                                                 self.sr)
        return (jnp.where(ok, p.astype(x.dtype), x), jnp.where(ok, mo, s["m"]), jnp.where(ok, vo, s["v"]))


# ------------------------------------------------------------------------------------- tree of increments, folded grad
# A branch leaf table W = (D + 1)^-1/2 sum_d levels[d][j >> (D - d)] is stored as its levels. Given dL/dW (2^D rows),
# the level gradients are its subtree sums times (D + 1)^-1/2. tree_norm and tree_update form them in registers from
# one read of dL/dW (leaves left to right, one open node per level), so the level gradients (twice the rows of W) are
# never written or read back.

INTERPRET = False  # run the tree kernels in the Pallas interpreter (CPU tests)


def _kernels():
    return INTERPRET or jax.default_backend() == "gpu"


TREE_COLS, TREE_WARPS = 1024, 4  # columns per program (at most), warps


def _tree_cols(F, cap=None):
    """Columns per program: a power of two <= cap dividing F with at least 64 programs, or None (XLA path: a narrow
    table would run its serial walk over the leaves on a few programs)."""
    b = cap or TREE_COLS
    while b >= 128 and (F % b or F // b < 64):
        b //= 2
    return b if b >= 128 else None


def _tree_walk(D, row_grad, visit, carry):
    """Leaves left to right in a loop, summed pairwise up the tree in the order of leaves.tree_sum's backward: left[d]
    holds the pending left child at level d. After leaf j, visit(d, row, g, done, carry) sees each level's node on
    the path; done says it is complete (it is False for the others: visit must leave memory and carry unchanged)."""
    def body(j, c):
        left, carry = c
        cur = row_grad(j)
        carry = visit(D, j, cur, True, carry)
        done, new = True, list(left)
        for d in range(D, 0, -1):
            right = ((j >> (D - d)) & 1) == 1
            new[d - 1] = jnp.where(done & ~right, cur, left[d - 1]) if done is not True else \
                jnp.where(right, left[d - 1], cur)
            cur, done = left[d - 1] + cur, right if done is True else done & right
            carry = visit(d - 1, j >> (D - d + 1), cur, done, carry)
        return new, carry
    zero = row_grad(0) * 0.0
    return jax.lax.fori_loop(0, 2**D, body, ([zero] * D, carry))[1]


def _tree_norm_kernel(g_ref, o_ref, *, D, bf):
    j = pl.program_id(0)
    cols = pl.ds(j * bf, bf)
    sc = (D + 1) ** -0.5

    def visit(d, r, g, done, acc):
        return acc + (jnp.square(g) if done is True else jnp.where(done, jnp.square(g), 0.0))
    o_ref[cols] = _tree_walk(D, lambda r: g_ref[r, cols].astype(jnp.float32) * sc, visit,
                             jnp.zeros((bf,), jnp.float32))


def tree_norm(G):
    """G (2^D, ...) -> sum over all levels of the squared level gradients."""
    D = int(np.log2(G.shape[0]))
    G2 = G.reshape(G.shape[0], -1)
    F = G2.shape[1]
    bf = _tree_cols(F)
    if bf is None or not _kernels():
        return sum(jnp.sum(jnp.square(g)) for g in _tree_grads(G))
    out = pl.pallas_call(functools.partial(_tree_norm_kernel, D=D, bf=bf), grid=(F // bf,),
                         in_specs=[_full(G2)], out_specs=pl.BlockSpec((F,), lambda j: (0,)),
                         out_shape=jax.ShapeDtypeStruct((F,), jnp.float32),
                         compiler_params=plgpu.CompilerParams(num_warps=TREE_WARPS, num_stages=1), interpret=INTERPRET)(G2)
    return jnp.sum(out)


def _tree_grads(G):
    """XLA reference: level gradients from dL/dW, as leaves.tree_sum's backward."""
    D = int(np.log2(G.shape[0]))
    g = G.astype(jnp.float32) * ((D + 1) ** -0.5)
    out = [g]
    for _ in range(D):
        g = g.reshape((g.shape[0] // 2, 2) + g.shape[1:]).sum(1)
        out.append(g)
    return out[::-1]


def _tree_update_kernel(s_ref, n_ref, g_ref, *refs, D, bf, F, b1, b2, eps, mu, nu, sr):
    L = D + 1
    ps, ms, vs = refs[:L], refs[L:2 * L], refs[2 * L:3 * L]
    po, mo, vo = refs[3 * L:4 * L], refs[4 * L:5 * L], refs[5 * L:]
    j = pl.program_id(0)
    cols = pl.ds(j * bf, bf)
    pick = lambda a, k: jnp.sum(jnp.where(jax.lax.broadcasted_iota(jnp.int32, a.shape, 0) == k, a, 0))
    s, seeds = s_ref[...], n_ref[...]
    scale, c1, c2, keep, ok = (pick(s, k) for k in range(5))
    col = (j * bf + jax.lax.broadcasted_iota(jnp.int32, (bf,), 0)).astype(jnp.uint32)
    sc = (D + 1) ** -0.5

    def visit(d, r, g, done, carry):
        def upd():
            seed = pick(seeds, d)
            gg = g * scale
            x, m0, v0 = ps[d][r, cols], ms[d][r, cols], vs[d][r, cols]
            m = b1 * m0.astype(jnp.float32) + (1 - b1) * gg
            v = b2 * v0.astype(jnp.float32) + (1 - b2) * gg * gg
            p = x.astype(jnp.float32) * keep - c1 * m / (jnp.sqrt(v) * c2 + eps)
            idx = jnp.asarray(r).astype(jnp.uint32) * jnp.uint32(F) + col
            mn, vn = _store(m, mu, seed, idx, sr), _store(v, nu, seed ^ jnp.uint32(0x5bd1e995), idx, sr)
            po[d][r, cols] = jnp.where(ok > 0, p.astype(x.dtype), x)
            mo[d][r, cols] = jnp.where(ok > 0, mn, m0)
            vo[d][r, cols] = jnp.where(ok > 0, vn, v0)
        upd() if done is True else pl.when(done)(upd)
        return carry
    _tree_walk(D, lambda r: g_ref[r, cols].astype(jnp.float32) * sc, visit, jnp.zeros((), jnp.float32))


def _full(a):
    return pl.BlockSpec(a.shape, lambda *_: (0,) * a.ndim)


def tree_update(levels, G, ms, vs, scale, c1, c2, keep, ok, count, ids, b1, b2, eps, mu, nu, sr):
    """FastAdam._dense on every level of one tree, the level gradients formed from G = dL/dW in registers.
    ids[d]: index of levels[d] in the flattened params (its rounding seed). -> new levels, ms, vs."""
    D = len(levels) - 1
    shp = [lv.shape for lv in levels]
    flat = lambda a: a.reshape(a.shape[0], -1)
    G2, P, M, V = flat(G), [flat(a) for a in levels], [flat(a) for a in ms], [flat(a) for a in vs]
    L = D + 1
    F = G2.shape[1]
    bf = _tree_cols(F)
    s = jnp.stack([jnp.asarray(a, jnp.float32) for a in (scale, c1, c2, keep, ok.astype(jnp.float32))] +
                  [jnp.zeros((), jnp.float32)] * 3)
    n = count.astype(jnp.uint32) * jnp.uint32(1000003) + jnp.uint32(7919) * (jnp.asarray(ids, jnp.uint32) + 1)
    n = jnp.zeros((max(8, 1 << (D + 1).bit_length()),), jnp.uint32).at[:L].set(n)  # rounding seed of each level
    args = [s, n, G2] + P + M + V
    out = pl.pallas_call(
        functools.partial(_tree_update_kernel, D=D, bf=bf, F=F, b1=b1, b2=b2, eps=eps, mu=mu, nu=nu, sr=sr),
        grid=(F // bf,), in_specs=[_full(a) for a in args], out_specs=[_full(a) for a in P + M + V],
        out_shape=[jax.ShapeDtypeStruct(a.shape, a.dtype) for a in P + M + V],
        input_output_aliases={3 + i: i for i in range(3 * L)},
        compiler_params=plgpu.CompilerParams(num_warps=TREE_WARPS, num_stages=1), interpret=INTERPRET)(*args)
    back = lambda a, i: a.reshape(shp[i])
    return ([back(a, i) for i, a in enumerate(out[:L])], [back(a, i) for i, a in enumerate(out[L:2 * L])],
            [back(a, i) for i, a in enumerate(out[2 * L:])])
