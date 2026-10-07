# usage: julia -t T run_scx.jl input outdir noversamples niter
using OnlinePCA, Serialization, Random
input, outdir = ARGS[1], ARGS[2]; ov, ni = parse(Int, ARGS[3]), parse(Int, ARGS[4])
mkpath(outdir); Random.seed!(1)
t_sumr = @elapsed scxsumr(input=input, outdir=outdir, scale="log", cper=1f4)
t_pca = @elapsed out = scxpca(input=input, scale="log", cper=1f4, dim=50, noversamples=ov, niter=ni,
    rowmeanlist=joinpath(outdir, "Feature_Means.csv"), colsumlist=joinpath(outdir, "Sample_NoCounts.csv"))
serialize(joinpath(outdir, "scx.jls"), (λ=out[2], U=out[3], t_sumr=t_sumr, t_pca=t_pca))
println("TIMES sumr=$(round(t_sumr,digits=1))s pca=$(round(t_pca,digits=1))s")
