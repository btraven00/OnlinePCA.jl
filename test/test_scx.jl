#####################################
println("####### scxchunks (scx cell-chunk stream) #######")
# Write a 10x HDF5 with HDF5.jl, stream it back through scx in uneven chunks,
# and check the reassembled genes × cells matrix is identical.
if isfile(OnlinePCA.libscx)
    let h5 = OnlinePCA.HDF5, ngenes = 60, ncells = 80
        Random.seed!(1)
        A = sprand(Float32, ngenes, ncells, 0.3)
        A.nzval .= rand(1:100, nnz(A))
        A[:, 5] .= 0; dropzeros!(A)  # an empty cell
        function write10x(f, A)
            h5.h5open(f, "w") do io
                m = h5.create_group(io, "matrix")
                m["barcodes"] = ["c$i" for i in 1:size(A, 2)]
                ft = h5.create_group(m, "features")
                ft["id"] = ["g$i" for i in 1:size(A, 1)]
                ft["name"] = ["G$i" for i in 1:size(A, 1)]
                m["data"] = Int32.(A.nzval)
                m["indices"] = Int32.(A.rowval .- 1)
                m["indptr"] = Int32.(A.colptr .- 1)
                m["shape"] = Int32[size(A)...]
            end
            f
        end
        f = write10x(joinpath(tmp, "scx_10x.h5"), A)
        @test scxshape(f) == (ngenes, ncells)
        chunks = collect(scxchunks(f; chunksize=7))
        @test first.(chunks) == [i:min(i + 6, ncells) for i in 1:7:ncells]
        @test reduce(hcat, [X for (_, X) in chunks]) == A

        # scxsumr: one pass gives the stats of the normalised matrix.
        nc, μ, σ2 = scxsumr(input=f, outdir=tmp, scale="sqrt", cper=10f0)
        An = sqrt.(10 .* Matrix(A) ./ sum(A, dims=1))
        An[:, 5] .= 0  # empty cell: 0/0 stays 0 in the sparse path
        @test nc ≈ vec(sum(A, dims=1))
        @test μ ≈ vec(mean(An, dims=2))
        @test σ2 ≈ vec(var(An, dims=2))

        # scxpca reproduces upstream tenxpca on the same file, lists and Ω.
        # Without the empty cell: tenxpca divides 0 by its zero library size.
        f = write10x(joinpath(tmp, "scx_10x_noempty.h5"), A[:, setdiff(1:ncells, 5)])
        scxsumr(input=f, outdir=tmp, scale="sqrt", cper=10f0)
        kw = (rowmeanlist=joinpath(tmp, "Feature_Means.csv"), colsumlist=joinpath(tmp, "Sample_NoCounts.csv"))
        Random.seed!(7)
        W, D, rm, rv, csv, N, M, TV, idp = OnlinePCA.tenxinit(f, 3, 5000, "matrix", kw.rowmeanlist, "",
            kw.colsumlist, nothing, nothing, nothing, 10f0, "sqrt", false)
        ref = OnlinePCA.tenxpca(f, nothing, "sqrt", kw.rowmeanlist, "", kw.colsumlist, 3, 5, 3, 5000,
            nothing, OnlinePCA.TENXPCA(), W, D, rm, rv, csv, N, M, TV, false, idp, "matrix", 10f0)
        Random.seed!(7)
        out = scxpca(; input=f, scale="sqrt", dim=3, chunksize=7, cper=10f0, kw...)
        @test out[2] ≈ ref[2] rtol = 1e-4
        @test abs.(out[3]' * ref[3]) ≈ I(3) atol = 1e-4  # same loadings up to sign
        @test out[5] ≈ ref[5] rtol = 1e-4
    end
else
    @info "skipping scxchunks test: $(OnlinePCA.libscx) not built (cargo build --release in deps/scx_capi)"
end
