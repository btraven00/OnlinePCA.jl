# Upstream bugs

Bugs in rikenbit/OnlinePCA.jl found while porting `tenxpca`/`tenxsumr` to
`scxpca`/`scxsumr` on this fork. They apply to upstream `master` at `bfb2f75`
and are not fixed here; `scxpca` avoids them. Candidates for upstream issues.

## 1. Keyword `tenxpca(...)` always throws `UndefVarError: pca`

`src/tenxpca.jl:43` passes a variable `pca` to the positional method, but the
keyword method never defines it (every other algorithm does, e.g.
`pca = HALKO()`). So the documented API, including the README example, fails
before reading any data:

```julia
tenxpca(tenxfile="Data.h5", scale="sqrt", rowmeanlist="Feature_SqrtMeans.csv",
        dim=3, chunksize=100, group="mm10")
# ERROR: UndefVarError: `pca` not defined
```

There is no `test/test_tenxpca.jl`, so CI does not catch it. Fix: add
`pca = TENXPCA()` to the keyword method. Workaround: call `OnlinePCA.tenxinit`
and the positional `OnlinePCA.tenxpca(..., OnlinePCA.TENXPCA(), ...)` directly
(as `test/test_scx.jl` does).

## 2. A cell with zero counts turns the whole result into NaN

`tenxnormalizex` (`src/tenxpca.jl:201`, `:204`) computes
`cper .* X ./ colsumvec'`. For a cell whose total count is 0, that column is
`0 / 0 = NaN`. The NaN goes into `XΩ` and `lu!` fails:

```
ArgumentError: matrix contains Infs or NaNs
```

Hit with a 10x file containing one empty barcode. Unfiltered Cell Ranger
output (`raw_feature_bc_matrix.h5`) is full of these. Fix: skip zero-sum
columns, or normalise only the nonzeros (what `scxnormalize!` does).

## 3. Omitting `colsumlist` divides by zero

When `colsumlist == ""`, `tenxinit` leaves `colsumvec` as zeros
(`src/tenxpca.jl:233`). `tenxnormalizex` divides by it anyway, so every entry
becomes `Inf` or `NaN`. The README example passes no `colsumlist`. Bug 1
currently hides this. Found by reading the code; not run. Fix: skip the
library-size step when no `colsumlist` is given, as `normalizex` in
`src/Utils.jl` already does.

## 4. 10x files without `matrix/shape` are rejected

`tenxnm` (`src/Utils.jl:100`) reads `<group>/shape`, which some 10x HDF5 files
lack. The Kang 2018 PBMC file (`pbmcs_ctrl.h5`) has only `barcodes`, `data`,
`features`, `indices` and `indptr`:

```
KeyError: key "matrix/shape" not found
```

Fix: fall back to `length(indptr) - 1` cells and `length(features/id)` (or
`genes`) features.

## 5. `loadchromium` fails on a gene chunk with no nonzeros

`src/Utils.jl:143` asserts `minimum(newindices) >= startp`. If no cell has a
count in genes `startp:endp`, `newindices` is empty and `minimum` throws
`ArgumentError`. This is likely after gene filtering with a small `chunksize`.
Found by reading the code; not run. Fix: guard the asserts with
`isempty(newindices)`.

## Performance note (not a bug)

`loadchromium` builds each gene chunk by reading every cell's slice of the 10x
file, so each pass scans the whole file once per gene chunk, about
`ngenes / chunksize` times, with two HDF5 reads per cell. On Kang ctrl
(6,548 cells × 14,053 genes, `dim=10`, `niter=3`), `tenxpca` takes 56 s and
`tenxsumr` 41 s. `scxpca` and `scxsumr` scan cell chunks once per pass and take
about 2 s each on the same data.
