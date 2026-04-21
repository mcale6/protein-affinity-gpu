# Tinygrad batched SASA — three adjustments

**Result on 1A2K (Apple Metal, 5 trials):** 126,437 ms → 2,381 ms end-to-end (**~53×**).
Numerical parity: tg ΔG = −9.1329 vs jax ΔG = −9.1869 (|Δ| = 0.054 kcal/mol).

Prior batched implementation materialized a 16k×16k interaction matrix upfront,
realized each block separately into a numpy buffer, and rebuilt the Metal kernel
from scratch for every block. The three changes below exploit tinygrad's lazy
graph without tripping the `graph_rewrite stack too big` error that chained-lazy
graphs hit on structures of this scale.

Commit: `653d1fa` (`Speed up tinygrad batched SASA on Metal — 126s → 2.4s on 1A2K`)
Follow-up: `bb0e021` (`Cache one TinyJit per (block, N, M) shape`) — required so
multi-structure runs can reuse the compiled kernel per shape instead of crashing
with `JitError: args mismatch in JIT`.

---

## 1. TinyJit the per-block kernel

**Before:** the per-block body was a plain Python function. Every iteration
rebuilt the Metal kernel from the lazy graph.

**After:** the block body is wrapped with `_TGTinyJit`. The first block pays
the compile cost; the remaining blocks hit the cache.

```python
def _sasa_block_tinygrad_impl(block_coords, block_radii, block_abs_idx,
                              all_coords, coords_norm2, all_radii_with_probe,
                              radii_probe_sq, sphere_points):
    ...
    return (n_points - buried_points).cast(_tg_dtypes.float32).realize()

_sasa_block_tinygrad = _TGTinyJit(_sasa_block_tinygrad_impl)
```

Per-shape cache (commit `bb0e021`) keyed on `(block_size, n_atoms, n_sphere_points)`:

```python
_sasa_block_jit_cache: dict[tuple[int, int, int], "_TGTinyJit"] = {}

def _get_sasa_block_jit(block_size, n_atoms, n_sphere_points):
    key = (block_size, n_atoms, n_sphere_points)
    jit = _sasa_block_jit_cache.get(key)
    if jit is None:
        jit = _TGTinyJit(_sasa_block_tinygrad_impl)
        _sasa_block_jit_cache[key] = jit
    return jit
```

### The `effective_start` trick

A naive loop gives the tail block a shorter shape, which triggers a *second*
kernel compile. We pull the tail window back so every call sees the same shape,
then slice the written output by `write_offset`:

```python
for start in range(0, n_atoms, block_size):
    end = min(start + block_size, n_atoms)
    effective_start = min(start, n_atoms - block_size)   # keep uniform shape
    block_coords = masked_coords[effective_start:effective_start + block_size] \
                       .contiguous().realize()
    ...
    block_out = sasa_block_jit(block_coords, ...)
    write_offset = start - effective_start
    block_slices.append(block_out[write_offset:write_offset + (end - start)])
```

### Why `.contiguous().realize()` on the slice

TinyJit rejects two input shapes:
- **Duplicate buffers.** A raw slice shares storage with the parent tensor;
  the JIT sees the same buffer address twice and errors out.
- **Const-folded scalars.** Anything computable at trace time gets baked into
  the compiled kernel and won't update on later calls. This is why
  `block_abs_idx` is a `[B]` int32 buffer rather than the integer `start`.

`.contiguous().realize()` gives the slice its own materialized buffer.

---

## 2. Pipelined cat over realized leaves

**Before:** per-block output was pulled to numpy on every iteration — the
compute graph died at the block boundary, killing any tinygrad fusion.

```python
# old
n_accessible = np.empty(n_atoms, dtype=np.float32)
for start in range(0, n_atoms, block_size):
    ...
    n_accessible[start:end] = (n_points - buried_points).numpy().astype(np.float32)
return (areas * _TGTensor(n_accessible) / n_points).realize()
```

**After:** each JIT call already returns a realized buffer, so `Tensor.cat`
stitches them as a **shallow 1-level graph over materialized inputs**. The
downstream `areas * (n_accessible / n_points)` then fuses across all blocks
without ever rebuilding a deep lazy graph.

```python
block_slices = []
for start in range(0, n_atoms, block_size):
    ...
    block_out = sasa_block_jit(...)          # realized [block_size]
    block_slices.append(block_out[write_offset:write_offset + (end - start)])

n_accessible = block_slices[0] if len(block_slices) == 1 \
               else _TGTensor.cat(*block_slices, dim=0)
```

**Why it doesn't trip `graph_rewrite stack too big`:** cat only recurses over
its direct inputs. With realized leaves, the pattern-matcher sees one level
and terminates. Chained-lazy cat (as in the single-pass SASA on large N) walks
through every op that produced each input and blows the recursion limit.

---

## 3. Inline the `[B, N]` interaction mask

**Before:** the full 16k×16k boolean interaction matrix was materialized
once upfront (~270 MB on 1A2K scale).

```python
# old
diff_inter = masked_coords[:, None, :] - masked_coords[None, :, :]
dist2_inter = (diff_inter ** 2).sum(axis=-1)
radsum2 = (radii_with_probe[:, None] + radii_with_probe[None, :]) ** 2
not_eye = _TGTensor.eye(coords.shape[0], dtype=_tg_dtypes.bool) == 0
interaction_matrix = ((dist2_inter <= radsum2) & not_eye).realize()   # 270 MB
```

**After:** each JIT call derives its own `[B, N]` mask in-kernel from the
`[B]` absolute-index buffer and the global coords:

```python
# inside _sasa_block_tinygrad_impl
block_norm2 = (block_coords * block_coords).sum(axis=-1)              # [B]
dot_bn = block_coords @ all_coords.transpose(-1, -2)                  # [B, N]
dist2_bn = block_norm2[:, None] + coords_norm2[None, :] - 2.0 * dot_bn

radsum = block_radii[:, None] + all_radii_with_probe[None, :]         # [B, N]
within = dist2_bn <= (radsum * radsum)
atom_idx = _TGTensor.arange(n_atoms, dtype=_tg_dtypes.int32)
not_self = atom_idx[None, :] != block_abs_idx[:, None]                # [B, N]
block_inter = within & not_self                                       # [B, N]
```

The `[B, N]` mask is hot inside the kernel (cache-resident) and never lands
in unified memory as a standalone tensor. Dropping the 270 MB realize frees
that bandwidth for the `[block, 100, N]` buried-points scratch that actually
wants the L2.

---

## Supporting change: block size = 768

Retuned `estimate_optimal_block_size` from 152 → 768 on Apple Metal based on
the throughput curve:

| block | 1A2K e2e (ms) |
|------:|--------------:|
|    64 |        26,824 |
|   152 |         9,317 |
|   256 |         6,824 |
|   512 |         3,537 |
|   768 |         2,284 |
|  1024 |         7,263 |

Past 768, the `[block, 100, N]` float32 scratch (~5 GB) spills out of the
Metal fast L2/MMU path.

```python
def estimate_optimal_block_size(n_atoms: int) -> int:
    return min(768, n_atoms)
```

---

## Supporting change: dot-product distance in contacts

Same identity `|a−b|² = |a|² + |b|² − 2⟨a,b⟩` replaces the `[N_t, N_b, 37, 37, 3]`
pairwise-diff tensor in `contacts.py`. Not one of the three SASA moves, but
bundled into the same commit because it's the same idea applied to the other
hot O(N²) kernel.

---

## What the changes do *not* fix

Heavy-benchmark results (Kahraman 2013 T3, 16 structures) show CPU beats
tinygrad on every structure in the set. Metal only wins past some atom-count
threshold that 1A2K crosses (53×) but these smaller/awkward structures don't.
The three adjustments are necessary but not sufficient; further speedup for
multi-structure runs likely needs atom14 compaction to shrink N before the
`[B, M, N]` probe-scatter kernel.

---

## 4. Block-mode buffer aliasing fix

**Symptom:** with the pipelined `Tensor.cat` of §2 in place, large structures
(2CFH, 2840 atoms) returned ~40% low SASA. Per-atom inspection showed values
flipping between 0 and 60 Å² — the block kernel itself was bit-identical to
the single-pass kernel when called once, but a multi-block run produced wrong
results.

**Root cause:** `_TGTinyJit` returns a `Tensor` that wraps a **persistent
output buffer**. Every call to the same JIT writes into the same address.
When we held three block outputs lazily and stitched them with `cat`, blocks 1
and 2 silently aliased to block 3's data — only block 0 retained its values
(it landed in a different first-call buffer slot).

**What didn't work:** `out + 0`, `out * 1`, `out.cast(out.dtype)`,
`Tensor.zeros + out[slice]`, `out.contiguous().realize()` — all of these
either become noops or collapse lazily back onto the aliased buffer.

**Fix:** detach each block to numpy at the moment it's produced, then rebuild
a single tensor for the trailing scaling. The outer accumulator is now a
numpy buffer rather than a chain of lazy slices, so subsequent JIT calls
can't overwrite earlier results.

```python
n_accessible_np = np.empty(n_atoms, dtype=np.float32)
for start, end, effective_start in _iter_blocks(n_atoms, block_size):
    block_coords = masked_coords[effective_start:effective_start + block_size] \
                       .contiguous().realize()
    block_radii = radii_with_probe[effective_start:effective_start + block_size] \
                       .contiguous().realize()
    block_abs_idx = _TGTensor(np.arange(effective_start,
                                        effective_start + block_size,
                                        dtype=np.int32)).realize()
    block_out = sasa_block_jit(block_coords, block_radii, block_abs_idx,
                               masked_coords, coords_norm2,
                               radii_with_probe, radii_probe_sq, sphere_points)
    write_offset = start - effective_start
    n_accessible_np[start:end] = block_out.numpy()[write_offset:write_offset + (end - start)]
n_accessible = _TGTensor(n_accessible_np)
areas = (4.0 * math.pi) * radii_probe_sq
return (areas * n_accessible / n_points).realize()
```

Cost: we lose §2's cross-block fusion of the trailing `areas * n / n_points`.
Verified bit-identical to the single-pass kernel across the full Kahraman set
at block sizes ∈ {2268, 1512, 1134, 768, 504, 256}.

---

## 5. Neighbor-cutoff mode (`mode="neighbor"`)

A third TinygradAdapter mode added alongside `"block"` (default) and
`"single"` (fully fused). Replaces the dense `[N, M, N]` buried-check
scratch with a `[N, M, K]` neighbor-only scratch via `Tensor.topk`.

```python
# inside _calculate_sasa_tinygrad_neighbor_impl
dist2_atom = atoms_norm2[:, None] + atoms_norm2[None, :] - 2.0 * (coords @ coords.T)
neighbor_idx = (-dist2_atom).topk(k_neighbors, dim=-1).indices    # [N, K] nearest
neighbor_coords = coords[neighbor_idx]                            # [N, K, 3]
neighbor_radii = radii_with_probe[neighbor_idx]                   # [N, K]
# probe-scatter then becomes [N, M, K] instead of [N, M, N]
```

Memory: at N=7394, K=64, M=100 the buried-check scratch is ~190 MB instead
of ~22 GB — the only viable option on memory-constrained GPUs (Colab T4 has
15 GB; the dense `single` kernel OOMs past ~5k atoms).

Speed (Apple Metal M-series, warm-mean ms, Kahraman 2013 T3, 16 structures):

| pdb  | N    | tg-block | tg-single | tg-neighbor | nei/blk |
|------|------|---------:|----------:|------------:|--------:|
| 2CFH | 2485 |    240.0 |     222.7 |       448.6 |   1.87× |
| 1MQ8 | 2840 |    344.5 |     252.5 |     2 277.5 |   6.61× |
| 1FQ1 | 3810 |    490.1 |     935.2 |     4 219.3 |   8.61× |
| 1ATN | 4929 |    848.8 |     659.9 |    11 370.9 |  13.40× |
| 1H1V | 5416 |    973.6 |     OOM¹  |       OOM¹  |       — |
| 1Y64 | 6119 |  1 153.2 |     901.8 |    22 131.9 |  19.19× |
| 1HE8 | 7394 |  1 630.0 |   1 794.8 |    20 235.5 |  12.41× |

¹ `tg-single` and `tg-neighbor` both raised
  `RuntimeError: Internal Error (0000000e:Internal Error)` on 1H1V — Metal
  OOM at warm runs (~10 GB scratch for `single`).

**Takeaway:** on Apple Metal `topk` is expensive enough that the saved scratch
doesn't pay for itself — `tg-neighbor` is **1.45–19.19× slower** than
`tg-block`, scaling badly with N. Its purpose is *memory*, not speed.
Reach for `mode="neighbor"` only when the dense modes OOM.

Numerical parity vs `tg-block` on the Kahraman set is within 0.001 kcal/mol
on 15/16 structures; 1MQ8 differs by 0.028 (K=64 misses one occluder for one
atom). Bump `k_neighbors` to recover it if needed.
