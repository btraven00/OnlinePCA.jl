#!/usr/bin/env bash
# usage: bench/run.sh NAME FILE   e.g. bench/run.sh tm ~/phd/data/tm_omnidata/tm-droplet/tm-droplet.h5ad
# scxpca at two configs (4 threads), then the converged in-memory reference (16 threads), then the comparison.
set -euo pipefail
name=$1; f=$2; B=$(cd "$(dirname "$0")" && pwd); O=$B/out
cd "$B/.."
for cfg in "5 3" "20 5"; do set -- $cfg
  /usr/bin/time -v julia --project=. -t 4 $B/run_scx.jl "$f" $O/$name-ov$1-ni$2 $1 $2 2>&1 | grep -E "TIMES|Maximum resident|Elapsed \(wall"
done
/usr/bin/time -v julia --project=. -t 16 $B/ref.jl "$f" $O/$name-ref.jls 2>&1 | grep --line-buffered -E "^it |kernel check|Maximum resident|Elapsed \(wall"
julia --project=. $B/compare.jl $O/$name-ref.jls $O/$name-ov5-ni3/scx.jls $O/$name-ov20-ni5/scx.jls
