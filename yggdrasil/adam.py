"""The trainer's Adam (train.build_optimizer) on a GPU, in place and in one pass per array.

build_optimizer is optax.apply_if_finite(optax.chain(clip_by_global_norm(grad_clip), optax.adam(schedule))), and
routed_lr_scale scales the routed layers' updates. step() computes that update in optax's order of operations and
returns the same optax state pytree, but each array is read and written once: no update arrays and no lax.cond copy
of the state. The HRE levels of an expert array take dL/dW of the per-expert array W (2^depth rows) and form their
gradients (subtree sums times (depth + 1)^-1/2, in experts.hre_sum's backward order) in registers while they are
updated, so the level gradients are never stored. Each array is split into 2^L0 subtrees, one program per column
block and subtree walking its experts left to right; the few levels above L0 are updated in XLA. Use inside a jit
that donates the params and the optimiser state.
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from optax._src.numerics import safe_increment


BF = 256  # columns per program
MIN_COLS = 1024  # narrower arrays (bias, decay) are small; XLA updates them
PROGRAMS = 1024  # target programs per array (column blocks x subtrees); 64 programs leave an L40S half idle
WARPS = 4
N_SCALARS = 8  # _adam's scalars: keep, norm, clip, bias corrections 1 and 2, -lr, routed_lr_scale, do_update


def _full(a):
    return pl.BlockSpec(a.shape, lambda *_: (0,) * a.ndim)


def _tree_walk(D, row_grad, visit, carry):
    """Experts left to right in a loop, summed pairwise up the tree in the order of hre_sum's backward: left[d]
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


def _tree_grads(G):
    """Level gradients from dL/dW in XLA, as hre_sum's backward."""
    D = int(np.log2(G.shape[0]))
    g = G.astype(jnp.float32) * ((D + 1) ** -0.5)
    out = [g]
    for _ in range(D):
        g = g.reshape((g.shape[0] // 2, 2) + g.shape[1:]).sum(1)
        out.append(g)
    return out[::-1]


def _keys(path):
    return [getattr(k, "key", getattr(k, "idx", k)) for k in path]


def _split(G2):
    """G2 (2^D, F) -> L0, the level whose 2^L0 subtrees the kernels walk one per program, or None (XLA path)."""
    R, F = G2.shape
    if F < MIN_COLS or F % BF:
        return None
    D = int(np.log2(R))
    return int(np.clip(np.ceil(np.log2(max(1, PROGRAMS // (F // BF)))), 0, D))


def _levels_above(g):
    """Level gradients 0 .. L0 - 1 from those of level L0, g (2^L0, F), summed pairwise as hre_sum's backward."""
    out = []
    while g.shape[0] > 1:
        g = g.reshape((g.shape[0] // 2, 2) + g.shape[1:]).sum(1)
        out.append(g)
    return out[::-1]


def _adam(p, g, m, v, s, b1, b2, eps, scale_lr):
    """One element of the update, in optax's order: clip (select), moments, bias correction, update, -lr,
    routed_lr_scale, p + u. Rejected steps (do_update = 0) keep p, m and v."""
    trigger, norm, clip, bc1, bc2, step, blr, ok = s
    g = jnp.where(trigger > 0, g, (g / norm) * clip)
    m2 = (1 - b1) * g + b1 * m
    v2 = (1 - b2) * (g * g) + b2 * v
    u = step * ((m2 / bc1) / (jnp.sqrt(v2 / bc2 + 0.0) + eps))
    if scale_lr:
        u = u * blr
    keep = ok > 0
    return jnp.where(keep, p + u, p), jnp.where(keep, m2, m), jnp.where(keep, v2, v)


def _norm_kernel(g_ref, sq_ref, bad_ref, top_ref, *, D, L0):
    """One column block of one subtree: sums of squares and non-finite counts of its level gradients (levels L0 .. D)
    -> sq, bad (2^L0, F); top (2^L0, F) = the subtree's level-L0 gradient."""
    j, sub = pl.program_id(0), pl.program_id(1)
    cols = pl.ds(j * BF, BF)
    Dl = D - L0

    def visit(dl, r, g, done, acc):
        sq, bad = jnp.square(g), jnp.where(jnp.isfinite(g), 0.0, 1.0)
        acc = (acc[0] + sq, acc[1] + bad) if done is True else \
            (acc[0] + jnp.where(done, sq, 0.0), acc[1] + jnp.where(done, bad, 0.0))
        if dl == 0:
            put = lambda: top_ref.__setitem__((sub, cols), g)
            put() if done is True else pl.when(done)(put)
        return acc
    z = jnp.zeros((BF,), jnp.float32)
    sq, bad = _tree_walk(Dl, lambda r: g_ref[sub * 2**Dl + r, cols].astype(jnp.float32) * (D + 1) ** -0.5, visit,
                         (z, z))
    sq_ref[sub, cols] = sq
    bad_ref[sub, cols] = bad


def tree_norm(G):
    """G = dL/dW (2^D, ...) -> (sum of squared level gradients, count of non-finite entries, level gradients
    0 .. L0 - 1 for tree_update, or None on the XLA path)."""
    D = int(np.log2(G.shape[0]))
    G2 = G.reshape(G.shape[0], -1)
    L0 = _split(G2)
    if L0 is None:
        gs = _tree_grads(G2)
        return sum(jnp.sum(jnp.square(g)) for g in gs), sum(jnp.sum(~jnp.isfinite(g)) for g in gs), None
    out = jax.ShapeDtypeStruct((2**L0, G2.shape[1]), jnp.float32)
    sq, bad, top = pl.pallas_call(
        functools.partial(_norm_kernel, D=D, L0=L0), grid=(G2.shape[1] // BF, 2**L0), in_specs=[_full(G2)],
        out_specs=[_full(out)] * 3, out_shape=[out] * 3,
        compiler_params=plgpu.CompilerParams(num_warps=WARPS, num_stages=1))(G2)
    above = _levels_above(top)
    return (jnp.sum(sq) + sum(jnp.sum(jnp.square(g)) for g in above),
            jnp.sum(bad) + sum(jnp.sum(~jnp.isfinite(g)) for g in above), above)


def _update_kernel(s_ref, g_ref, *refs, D, L0, b1, b2, eps, scale_lr):
    """One column block of one subtree: _adam on its levels L0 .. D, the level gradients formed from g = dL/dW.
    refs: p, m, v of levels L0 .. D (inputs), then the same as aliased outputs."""
    L = D + 1 - L0
    ps, ms, vs, po, mo, vo = (refs[i * L:(i + 1) * L] for i in range(6))
    j, sub = pl.program_id(0), pl.program_id(1)
    cols = pl.ds(j * BF, BF)
    sv = s_ref[...]
    s = tuple(jnp.sum(jnp.where(jax.lax.broadcasted_iota(jnp.int32, sv.shape, 0) == k, sv, 0.0))
              for k in range(N_SCALARS))
    Dl = D - L0

    def visit(dl, r, g, done, carry):
        row = sub * 2**dl + r

        def upd():
            po[dl][row, cols], mo[dl][row, cols], vo[dl][row, cols] = _adam(
                ps[dl][row, cols], g, ms[dl][row, cols], vs[dl][row, cols], s, b1, b2, eps, scale_lr)
        upd() if done is True else pl.when(done)(upd)
        return carry
    _tree_walk(Dl, lambda r: g_ref[sub * 2**Dl + r, cols].astype(jnp.float32) * (D + 1) ** -0.5, visit,
               jnp.zeros((), jnp.float32))


def tree_update(levels, G, ms, vs, s, above, b1, b2, eps, scale_lr):
    """_adam on every level of one tree, G = dL/dW of its folded table, above = tree_norm's levels above L0.
    -> new levels, ms, vs."""
    D = len(levels) - 1
    shp = [lv.shape for lv in levels]
    flat = lambda a: a.reshape(a.shape[0], -1)
    G2, P, M, V = flat(G), [flat(a) for a in levels], [flat(a) for a in ms], [flat(a) for a in vs]
    sl = tuple(s[k] for k in range(N_SCALARS))
    L0 = _split(G2)
    xla = lambda gs, n: [_adam(p, g, m, v, sl, b1, b2, eps, scale_lr) for p, g, m, v in zip(P[:n], gs, M[:n], V[:n])]
    if L0 is None:
        out = xla(_tree_grads(G2), D + 1)
    else:
        out = xla(above, L0)
        L = D + 1 - L0
        args = [s, G2] + P[L0:] + M[L0:] + V[L0:]
        res = pl.pallas_call(
            functools.partial(_update_kernel, D=D, L0=L0, b1=b1, b2=b2, eps=eps, scale_lr=scale_lr),
            grid=(G2.shape[1] // BF, 2**L0), in_specs=[_full(a) for a in args], out_specs=[_full(a) for a in args[2:]],
            out_shape=[jax.ShapeDtypeStruct(a.shape, a.dtype) for a in args[2:]],
            input_output_aliases={2 + i: i for i in range(3 * L)},
            compiler_params=plgpu.CompilerParams(num_warps=WARPS, num_stages=1))(*args)
        out += list(zip(res[:L], res[L:2 * L], res[2 * L:]))
    return tuple([o[k].reshape(shp[i]) for i, o in enumerate(out)] for k in range(3))


def step(params, grads, state, lr, clip, max_errors, routed_lr_scale=1.0, b1=0.9, b2=0.999, eps=1e-8):
    """-> (params, state, global gradient norm before clipping). grads matches params, except that each HRE array
    {"hre_levels": {k: [level 0..depth]}} comes as dL/dW of its per-expert array under k (experts.materialize)."""
    adam, sch = state.inner_state[1]
    flat, tdef = jax.tree_util.tree_flatten_with_path(params)
    paths, ps = [p for p, _ in flat], [x for _, x in flat]
    mus, nus = tdef.flatten_up_to(adam.mu), tdef.flatten_up_to(adam.nu)
    trees, gs = {}, []  # (path to the layer dict, key) -> [(level, index in flat)]
    for i, path in enumerate(paths):
        keys = _keys(path)
        if len(keys) >= 3 and keys[-3] == "hre_levels":
            trees.setdefault((tuple(keys[:-3]), keys[-2]), []).append((keys[-1], i))
            gs.append(None)
            continue
        g = grads
        for k in keys:
            g = g[k]
        gs.append(g)
    tree_g = {}
    for (pre, k), lv in trees.items():
        g = grads
        for kk in list(pre) + [k]:
            g = g[kk]
        tree_g[(pre, k)] = (g, [i for _, i in sorted(lv)])
    sq = sum(jnp.sum(jnp.square(g)) for g in gs if g is not None)
    bad = sum(jnp.sum(~jnp.isfinite(g)) for g in gs if g is not None)
    above = {}
    for key, (G, _) in tree_g.items():
        s2, b2_, above[key] = tree_norm(G)
        sq, bad = sq + s2, bad + b2_
    norm = jnp.sqrt(sq)
    isfinite = bad == 0
    notfinite = jnp.where(isfinite, jnp.zeros([], jnp.int32), safe_increment(state.notfinite_count))
    ok = jnp.logical_or(isfinite, notfinite > max_errors)
    count_inc = safe_increment(adam.count)
    f32 = lambda a: jnp.asarray(a, jnp.float32)

    def scalars(routed):
        return jnp.stack([f32(norm < clip), f32(norm), f32(clip), f32(1 - b1**count_inc), f32(1 - b2**count_inc),
                          f32(-1 * lr(sch.count)), f32(routed_lr_scale if routed else 1.0), f32(ok)])
    new_p, new_m, new_v = list(ps), list(mus), list(nus)
    for (pre, k), (G, ids) in tree_g.items():
        br = "routed_layers" in pre
        out = tree_update([ps[i] for i in ids], G, [mus[i] for i in ids], [nus[i] for i in ids], scalars(br),
                          above[(pre, k)], b1, b2, eps, br and routed_lr_scale != 1.0)
        for i, a, m, v in zip(ids, *out):
            new_p[i], new_m[i], new_v[i] = a, m, v
    for i, g in enumerate(gs):
        if g is not None:
            br = "routed_layers" in _keys(paths[i])
            new_p[i], new_m[i], new_v[i] = _adam(ps[i], g, mus[i], nus[i], tuple(scalars(br)), b1, b2, eps,
                                                 br and routed_lr_scale != 1.0)
    un = lambda xs: jax.tree_util.tree_unflatten(tdef, xs)
    new_adam = adam._replace(count=jnp.where(ok, count_inc, adam.count), mu=un(new_m), nu=un(new_v))
    new_sch = sch._replace(count=jnp.where(ok, safe_increment(sch.count), sch.count))
    return un(new_p), state._replace(
        notfinite_count=notfinite, last_finite=isfinite,
        total_notfinite=jnp.where(isfinite, state.total_notfinite, safe_increment(state.total_notfinite)),
        inner_state=(state.inner_state[0], (new_adam, new_sch))), norm
