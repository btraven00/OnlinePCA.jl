# Converged reference: same normalisation in memory, implicit centring, block subspace iteration.
using OnlinePCA, SparseArrays, LinearAlgebra, Serialization, Random
include(joinpath(@__DIR__, "kern.jl"))
function main(input, out)
    cols = SparseMatrixCSC{Float32,Int}[]
    for (cells, C) in scxchunks(input; chunksize=5000)
        cs = vec(sum(C, dims=1)); OnlinePCA.scxnormalize!(C, "log", 1f4, cs, Threads.nthreads()); push!(cols, C)
    end
    X = reduce(hcat, cols); empty!(cols); GC.gc()
    N, M = size(X); μ = vec(sum(X, dims=2)) ./ M
    parts = collect(Iterators.partition(1:M, cld(M, 4 * Threads.nthreads())))
    # kernel check against SparseArrays on 3 columns
    G3 = randn(N, 3); Z3 = randn(M, 3)
    e1 = norm(AtG(X, μ, parts, G3) - (X' * G3 .- (μ' * G3))) / norm(X' * G3)
    e2 = norm(AZ(X, μ, parts, Z3) - (X * Z3 .- μ * sum(Z3, dims=1))) / norm(X * Z3)
    println("kernel check rel err AtG=$e1 AZ=$e2"); flush(stdout)
    Random.seed!(0); l = 100; t0 = time()
    Q = Matrix(qr(randn(N, l)).Q); λold = zeros(50)
    for it in 1:80
        Q = Matrix(qr(AZ(X, μ, parts, AtG(X, μ, parts, Q))).Q)
        if it % 5 == 0
            F = svd(AtG(X, μ, parts, Q)'); λ = F.S[1:50] .^ 2 ./ M
            d = maximum(abs.(λ .- λold) ./ λ); λold = λ
            println("it $it max rel Δλ = $d  t=$(round(time()-t0))s"); flush(stdout)
            if d < 1e-10 || it == 80
                serialize(out, (λ=λ, U=(Q*F.U)[:, 1:50], it=it, d=d)); break
            end
        end
    end
end
main(ARGS[1], ARGS[2])
