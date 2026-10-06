#####################################
println("####### scxchunks (scx cell-chunk stream) #######")
# Write a 10x HDF5 with HDF5.jl, stream it back through scx in uneven chunks,
# and check the reassembled genes × cells matrix is identical.
if isfile(OnlinePCA.libscx)
    let h5 = OnlinePCA.HDF5, ngenes = 30, ncells = 23
        Random.seed!(1)
        A = sprand(Float32, ngenes, ncells, 0.2)
        A.nzval .= rand(1:100, nnz(A))
        A[:, 5] .= 0; dropzeros!(A)  # an empty cell
        f = joinpath(tmp, "scx_10x.h5")
        h5.h5open(f, "w") do io
            m = h5.create_group(io, "matrix")
            m["barcodes"] = ["c$i" for i in 1:ncells]
            ft = h5.create_group(m, "features")
            ft["id"] = ["g$i" for i in 1:ngenes]
            ft["name"] = ["G$i" for i in 1:ngenes]
            m["data"] = Int32.(A.nzval)
            m["indices"] = Int32.(A.rowval .- 1)
            m["indptr"] = Int32.(A.colptr .- 1)
            m["shape"] = Int32[ngenes, ncells]
        end
        @test scxshape(f) == (ngenes, ncells)
        chunks = collect(scxchunks(f; chunksize=7))
        @test [c for (c, _) in chunks] == [1:7, 8:14, 15:21, 22:23]
        @test reduce(hcat, [X for (_, X) in chunks]) == A
    end
else
    @info "skipping scxchunks test: $(OnlinePCA.libscx) not built (cargo build --release in deps/scx_capi)"
end
