# Products with the centred operator A = X - μ1' (genes × cells), k innermost on l × n buffers.
function AtG(X, μ, parts, G)               # returns A'G as M × l
    l = size(G, 2); Gt = permutedims(G); Zt = zeros(l, size(X, 2)); c = vec(μ' * G)
    Threads.@threads for p in parts
        @inbounds for j in p
            for i in nzrange(X, j)
                v = X.nzval[i]; r = X.rowval[i]
                @simd for k in 1:l; Zt[k, j] += v * Gt[k, r]; end
            end
            @simd for k in 1:l; Zt[k, j] -= c[k]; end
        end
    end
    permutedims(Zt)
end
function AZ(X, μ, parts, Z)                # returns A Z as N × l
    l = size(Z, 2); Zt = permutedims(Z); bufs = [zeros(l, size(X, 1)) for _ in parts]
    Threads.@threads for t in eachindex(parts)
        B = bufs[t]
        @inbounds for j in parts[t], i in nzrange(X, j)
            v = X.nzval[i]; r = X.rowval[i]
            @simd for k in 1:l; B[k, r] += v * Zt[k, j]; end
        end
    end
    permutedims(sum(bufs)) .- μ * sum(Z, dims=1)
end
