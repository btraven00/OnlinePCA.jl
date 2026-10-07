using Serialization, LinearAlgebra, Printf
r = deserialize(ARGS[1])
for f in ARGS[2:end]
    s = deserialize(f); U = Float64.(s.U); λ = Float64.(s.λ)
    rel = abs.(λ .- r.λ) ./ r.λ
    cosk = [abs(dot(U[:, k], r.U[:, k])) for k in 1:50]
    ov(k) = norm(r.U[:, 1:k]' * U[:, 1:k])^2 / k   # subspace overlap, 1 = identical
    @printf("%s\n  time sumr %.1fs pca %.1fs\n  λ rel err: PC1-10 max %.1e | PC11-30 max %.1e | PC31-50 max %.1e\n",
        f, s.t_sumr, s.t_pca, maximum(rel[1:10]), maximum(rel[11:30]), maximum(rel[31:50]))
    @printf("  per-PC |cos|: min PC1-10 %.4f | PC11-30 %.4f | PC31-50 %.4f ; first PC with |cos|<0.99: %s\n",
        minimum(cosk[1:10]), minimum(cosk[11:30]), minimum(cosk[31:50]), something(findfirst(<(0.99), cosk), "none"))
    @printf("  subspace overlap k=10 %.5f  k=30 %.5f  k=50 %.5f\n", ov(10), ov(30), ov(50))
end
