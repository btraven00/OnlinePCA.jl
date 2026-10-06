# Cell-chunk streaming of any scx-readable file (h5ad, 10x h5, h5seurat, mtx, …)
# via the scx_capi cdylib in deps/scx_capi.
#
# Each iteration yields `(cells, X)`: `cells` is the range of cell indices in
# the chunk and `X` is a genes × cells `SparseMatrixCSC{Float32,Int32}`, i.e.
# the same orientation OnlinePCA uses (rows = genes, columns = cells).

# ponytail: local build path; swap for a JLL product once this is registered.
const libscx = get(ENV, "SCX_CAPI_LIB",
    joinpath(@__DIR__, "..", "deps", "scx_capi", "target", "release", "libscx_capi.so"))

scxerror() = error("scx: " * unsafe_string(ccall((:scx_last_error, libscx), Cstring, ())))

struct ScxChunks
    path::String
    chunksize::Int
end

"""
    scxchunks(path; chunksize=5000)

Stream the count matrix of `path` as genes × cells sparse chunks of up to
`chunksize` cells, in one sequential pass. Iterate with
`for (cells, X) in scxchunks(path) ... end`.
"""
scxchunks(path::AbstractString; chunksize::Integer=5000) = ScxChunks(String(path), Int(chunksize))

"""
    scxshape(path) -> (ngenes, ncells)

Matrix shape in OnlinePCA orientation, without streaming any data.
"""
function scxshape(path::AbstractString)
    h = scxopen(path, 1)
    try
        nobs, nvars = Ref{Csize_t}(0), Ref{Csize_t}(0)
        ccall((:scx_shape, libscx), Cvoid, (Ptr{Cvoid}, Ref{Csize_t}, Ref{Csize_t}), h, nobs, nvars)
        return Int(nvars[]), Int(nobs[])
    finally
        ccall((:scx_close, libscx), Cvoid, (Ptr{Cvoid},), h)
    end
end

function scxopen(path, chunksize)
    h = ccall((:scx_open, libscx), Ptr{Cvoid}, (Cstring, Csize_t), path, chunksize)
    h == C_NULL && scxerror()
    h
end

Base.IteratorSize(::Type{ScxChunks}) = Base.SizeUnknown()
Base.eltype(::Type{ScxChunks}) = Tuple{UnitRange{Int},SparseMatrixCSC{Float32,Int32}}

function Base.iterate(it::ScxChunks)
    h = scxopen(it.path, it.chunksize)
    nobs, nvars = Ref{Csize_t}(0), Ref{Csize_t}(0)
    ccall((:scx_shape, libscx), Cvoid, (Ptr{Cvoid}, Ref{Csize_t}, Ref{Csize_t}), h, nobs, nvars)
    iterate(it, (h, Int(nvars[])))
end

function Base.iterate(::ScxChunks, (h, ngenes))
    off, nrows, nnz = Ref{Csize_t}(0), Ref{Csize_t}(0), Ref{Csize_t}(0)
    rc = ccall((:scx_next, libscx), Cint,
        (Ptr{Cvoid}, Ref{Csize_t}, Ref{Csize_t}, Ref{Csize_t}), h, off, nrows, nnz)
    if rc != 1
        # ponytail: the handle is only freed when the loop runs to the end (or
        # errors); a `break` leaks it and its reader thread until process exit.
        ccall((:scx_close, libscx), Cvoid, (Ptr{Cvoid},), h)
        rc == 0 ? (return nothing) : scxerror()
    end
    colptr = Vector{Int32}(undef, nrows[] + 1)
    rowval = Vector{Int32}(undef, nnz[])
    nzval = Vector{Float32}(undef, nnz[])
    ccall((:scx_copy, libscx), Cvoid,
        (Ptr{Cvoid}, Ptr{Int32}, Ptr{Int32}, Ptr{Float32}), h, colptr, rowval, nzval)
    cells = (off[] + 1):(off[] + nrows[])
    X = SparseMatrixCSC{Float32,Int32}(ngenes, Int(nrows[]), colptr, rowval, nzval)
    ((cells, X), (h, ngenes))
end
