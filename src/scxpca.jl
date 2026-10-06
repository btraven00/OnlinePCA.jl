# Cell-major counterparts of tenxsumr/tenxpca on top of `scxchunks`.
#
# tenxpca streams gene chunks out of a cell-major (CSC) 10x file, so every
# gene chunk rescans the whole file: N/chunksize full scans per pass. Here
# every pass is a single sequential scan of cell chunks, which also works for
# any format scx reads (h5ad, 10x h5, h5seurat, mtx, zarr, ...).
#
# Products are accumulated per cell chunk C = X[:, cells] (genes × cells):
#   X Ω   = Σ C Ω[cells, :]        X' L = [C' L ; ...]        Q' X = [Q' C  ...]
# Centering and row scaling are rank-1 / diagonal corrections, applied after
# each pass exactly as tenxpca does.

# Library-size normalisation + variance stabilisation of one cell chunk, in
# place on the nonzeros (both transforms map 0 to 0, so X stays sparse).
# Same transform as tenxnormalizex, except that with no `colsum` the
# library-size step is skipped instead of dividing by zero.
function scxnormalize!(X::SparseMatrixCSC, scale, cper, colsum, ntasks)
    f = scale == "sqrt" ? sqrt : scale == "log" ? (x -> log10(x + 1)) : identity
    _normalize!(X, f, cper, colsum, ntasks)
end
function _normalize!(X, f::F, cper, colsum, ntasks) where {F}
    v = X.nzval
    tforeach(size(X, 2), ntasks) do j
        @inbounds for k in SparseArrays.nzrange(X, j)
            v[k] = f(colsum === nothing ? v[k] : cper * v[k] / colsum[j])
        end
    end
    X
end

# Default task count: 4 matched 8 and 16 threads on a 160k-cell pass (reading
# is the floor beyond that) with less accumulator memory. Julia's thread count
# is fixed at startup, so this can only cap it, not raise it.
function defaultthreads()
    Threads.nthreads() == 1 && @info "scx: Julia is single-threaded; start it with `julia -t 4` to speed up scxpca" maxlog = 1
    min(4, Threads.nthreads())
end

# Run f(j) for j in 1:n on `ntasks` tasks, each over a contiguous block of j.
function tforeach(f::F, n, ntasks) where {F}
    Threads.@threads for part in collect(Iterators.partition(1:n, cld(max(n, 1), ntasks)))
        for j in part
            f(j)
        end
    end
end

# Multithreaded sparse × dense kernels for one chunk X (genes × cells). The
# dense operands and results are kept transposed (components × ·) so the inner
# loop over the l components is contiguous.

# accs[t] .+= (X * B[cells, :])' for Bt = B'. Each of the length(accs) tasks
# takes a block of cells and accumulates into its own l × genes buffer;
# sum(accs) after a pass.
function mul_acc!(accs, X, Bt, cells)
    parts = collect(Iterators.partition(1:size(X, 2), cld(size(X, 2), length(accs))))
    Threads.@threads for t in eachindex(parts)
        acc = accs[t]
        @inbounds for j in parts[t], k in SparseArrays.nzrange(X, j)
            r, v, b = X.rowval[k], X.nzval[k], cells[j]
            @simd for c in axes(acc, 1)
                acc[c, r] += v * Bt[c, b]
            end
        end
    end
end

# Zt[:, cells] = (X' * L)' for Lt = L'. Each cell's column is independent.
function tmul!(Zt, X, Lt, cells, ntasks)
    tforeach(size(X, 2), ntasks) do j
        z = view(Zt, :, cells[j])
        fill!(z, 0)
        @inbounds for k in SparseArrays.nzrange(X, j)
            r, v = X.rowval[k], X.nzval[k]
            @simd for c in eachindex(z)
                z[c] += v * Lt[c, r]
            end
        end
    end
end

"""
    scxsumr(; input, outdir=".", scale="sqrt", cper=1f0, chunksize=5000, threads=min(4, Threads.nthreads()))

Per-feature mean and variance of the normalised matrix that `scxpca` will
decompose with the same `scale`/`cper`, in one pass over `input`.

Writes `Sample_NoCounts.csv` (raw counts per cell), `Feature_Means.csv` and
`Feature_Vars.csv` (sample variance) to `outdir`, and returns them.
"""
function scxsumr(; input::AbstractString="", outdir::Union{Nothing,AbstractString}=".",
    scale::AbstractString="sqrt", cper::Number=1.0f0, chunksize::Number=5000,
    threads::Integer=defaultthreads())
    scale in ("sqrt", "log", "raw") || error("scale must be specified as sqrt, log, or raw")
    N, M = scxshape(input)
    nc = zeros(Float64, M)
    s = zeros(Float64, N)
    ss = zeros(Float64, N)
    # Each chunk holds every gene of its cells, so a cell's count total is
    # known before the cell is normalised: one pass, no second scan.
    for (cells, X) in scxchunks(input; chunksize=chunksize)
        colsum = vec(sum(X, dims=1))
        nc[cells] .= colsum
        scxnormalize!(X, scale, cper, colsum, threads)
        for k in eachindex(X.nzval)
            v = Float64(X.nzval[k])
            s[X.rowval[k]] += v
            ss[X.rowval[k]] += v * v
        end
    end
    # ponytail: sum/sum-of-squares in Float64; switch to Welford if a dataset
    # ever shows cancellation (normalised count data does not).
    means = s ./ M
    vars = (ss .- M .* means .^ 2) ./ (M - 1)
    if outdir isa String
        write_csv(joinpath(outdir, "Sample_NoCounts.csv"), nc)
        write_csv(joinpath(outdir, "Feature_Means.csv"), means)
        write_csv(joinpath(outdir, "Feature_Vars.csv"), vars)
    end
    return nc, means, vars
end

"""
    scxpca(; input, outdir=nothing, scale="sqrt", rowmeanlist="", rowvarlist="",
           colsumlist="", dim=3, noversamples=5, niter=3, chunksize=5000, cper=1f0,
           threads=min(4, Threads.nthreads()))

The randomized SVD of `tenxpca`, streaming cell chunks from any scx-readable
`input` with one sequential scan per pass. Rows are genes, columns are cells.
`rowmeanlist`, `rowvarlist` and `colsumlist` are the CSVs written by `scxsumr`.

`threads` caps the tasks used for the sparse products; Julia's own thread
count is set at startup (`julia -t 4`).

Returns `(V, λ, U, Scores, ExpVar, TotalVar)` as `tenxpca` does.
"""
function scxpca(; input::AbstractString="", outdir::Union{Nothing,AbstractString}=nothing,
    scale::AbstractString="sqrt", rowmeanlist::AbstractString="", rowvarlist::AbstractString="",
    colsumlist::AbstractString="", dim::Number=3, noversamples::Number=5, niter::Number=3,
    chunksize::Number=5000, cper::Number=1.0f0, threads::Integer=defaultthreads())
    scale in ("sqrt", "log", "raw") || error("scale must be specified as sqrt, log, or raw")
    N, M = scxshape(input)
    l = dim + noversamples
    @assert 0 < dim ≤ l ≤ min(N, M)
    @assert niter ≥ 1  # Q comes from the last power iteration
    μ = rowmeanlist == "" ? zeros(Float32, N) : vec(read_csv(rowmeanlist, Float32))
    σ2 = rowvarlist == "" ? nothing : vec(read_csv(rowvarlist, Float32))
    colsum = colsumlist == "" ? nothing : vec(read_csv(colsumlist, Float32))
    # Row scaling by 1/variance (as tenxpca) is a diagonal factor on the left.
    μs = σ2 === nothing ? μ : μ ./ σ2
    scalerows(Y) = σ2 === nothing ? Y : Y ./ σ2

    # One normalised pass over the data; `f(cells, X)` sees each chunk.
    function pass(f)
        for (cells, X) in scxchunks(input; chunksize=chunksize)
            f(cells, scxnormalize!(X, scale, cper, colsum === nothing ? nothing : view(colsum, cells), threads))
        end
    end

    # One l × N accumulator per task (4 tasks, 28k genes, l=15: 7 MB).
    newaccs() = [zeros(Float32, l, N) for _ in 1:threads]

    Ω = rand(Float32, M, l)
    println("Random Projection : Y = A Ω")
    accs, Ωt = newaccs(), permutedims(Ω)
    TotalVar = 0.0
    pass() do cells, X
        mul_acc!(accs, X, Ωt, cells)
        TotalVar += sum(abs2, X.nzval)  # uncentred, unscaled: tenxinit's tv()
    end
    XΩ = permutedims(sum(accs))
    TotalVar /= M
    Y = scalerows(XΩ) .- μs .* sum(Ω, dims=1)
    L = lu!(Y).L
    local Q

    for i in 1:niter
        println("##### " * string(i) * " / " * string(niter) * " niter #####")
        println("Normalized power iterations (1/3) : A' L")
        AtLt = zeros(Float32, l, M)
        Lst = permutedims(scalerows(L))
        pass() do cells, X
            tmul!(AtLt, X, Lst, cells, threads)
        end
        AtL = permutedims(AtLt) .- μs' * L

        println("Normalized power iterations (2/3) : A A' L")
        accs, AtLt = newaccs(), permutedims(AtL)
        pass() do cells, X
            mul_acc!(accs, X, AtLt, cells)
        end
        XAtL = permutedims(sum(accs))
        Y = scalerows(XAtL) .- μs .* sum(AtL, dims=1)
        if i < niter
            println("Normalized power iterations (3/3) : L = lu(A A' L)")
            L = lu!(Y).L
        else
            println("QR factorization  (3/3) : Q = qr(A A' L)")
            Q = Array(qr!(Y).Q)
        end
    end

    println("Calculation of small matrix : B = Q' A")
    QtX = zeros(Float32, l, M)
    Qst = permutedims(scalerows(Q))
    pass() do cells, X
        tmul!(QtX, X, Qst, cells, threads)
    end
    B = QtX .- Q' * μs

    println("SVD with small matrix : svd(B)")
    W, σ, V = svd(B)
    U = Q * W
    λ = σ .* σ ./ M
    Scores = V[:, 1:dim] .* λ[1:dim]'
    ExpVar = sum(λ) / TotalVar
    out = (V[:, 1:dim], λ[1:dim], U[:, 1:dim], Scores, ExpVar, TotalVar)
    if outdir isa String
        write_csv(joinpath(outdir, "Eigen_vectors.csv"), out[1])
        write_csv(joinpath(outdir, "Eigen_values.csv"), out[2])
        write_csv(joinpath(outdir, "Loadings.csv"), out[3])
        write_csv(joinpath(outdir, "Scores.csv"), out[4])
        write_csv(joinpath(outdir, "ExpVar.csv"), out[5])
        write_csv(joinpath(outdir, "TotalVar.csv"), out[6])
    end
    return out
end
