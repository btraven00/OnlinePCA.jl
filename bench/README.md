# scxpca accuracy and cost

`run.sh NAME FILE` runs `scxsumr` + `scxpca` (dim 50, `scale="log"`, `cper=1e4`, all genes,
4 threads) at the default (`noversamples=5, niter=3`) and a tuned (`20, 5`) config, then
`ref.jl`: the same normalised matrix in memory, implicitly centred, block subspace
iteration (l=100) run until the top-50 eigenvalues change by <1e-10. `kern.jl` holds its
sparse kernels, checked against SparseArrays on every run. `compare.jl` reports
eigenvalue error, per-PC |cos| and subspace overlap ‖U_refᵀU‖²_F / k.

## Results, 2026-10-07 (16-core laptop, 112877d)

| | wall | peak RSS | overlap k=10 / 30 / 50 |
|---|---|---|---|
| tm-droplet h5ad (23,433 × 54,865), default | 41 s | 1.35 GB | 0.99996 / 0.997 / 0.939 |
| tm-droplet, tuned | 52 s | 1.42 GB | 0.99997 / 0.999 / 0.990 |
| pbmc_multimodal h5seurat SCT/counts (20,729 × 161,764), default | 4:08 | 1.42 GB | 0.9998 / 0.980 / 0.800 |
| pbmc_multimodal, tuned | 7:15 | 1.45 GB | 0.9998 / 0.993 / 0.898 |
| in-memory reference, tm / pbmc | 4:34 / 16:16 | 6.2 / 15.0 GB | — |

- Memory is flat in cell count. Accuracy past ~PC20 depends on the spectrum: full-gene
  pbmc has small trailing eigengaps, and more passes help without closing the gap.
- PC1-10 eigenvalues are off by ~8e-4 on pbmc in both configs. Unverified: likely the
  Float32 accumulators, not the iteration count.
- A pass over the pbmc h5seurat costs ~30 s against ~4 s for tm's h5ad: it is deflate
  compressed, chunked at 1024/2048 elements, and stores Float64 values. Not measured:
  converting to uncompressed Float32 h5ad first.
