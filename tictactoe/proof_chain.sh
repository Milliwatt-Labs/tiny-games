#!/bin/bash
# H=3 proof chain (run detached; work files go in $WORK, default ./proof):
#   1. cube-and-conquer of cnf/h3p.cnf (cnfproof.py conquer; checkpointed, resumes on restart)
#   2. cover check (F and not(every leaf cube) is UNSAT, LRAT checked)
#   3. if both pass (H=3 UNSAT): fewest nonzero weights at H=4, nonzero <= 46 then <= 45 (portfolio, LRAT)
# Markers in runs/h3-cnc/: CHAIN-H3-UNSAT, CHAIN-H3-SAT, CHAIN-FAILED; chain.json holds the stage.
#
#   mkdir -p proof && nohup bash proof_chain.sh > proof/chain.log 2>&1 < /dev/null &
set -u
CODE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "${WORK:-proof}"
cd "${WORK:-proof}"
mkdir -p cnf runs
PY=${PY:-python3}
CNC=runs/h3-cnc
WORKERS=${WORKERS:-14}
mkdir -p "$CNC"
echo $$ > "$CNC/chain.pid"
stage() { printf '{"stage": "%s", "updatedAt": "%s", "pid": %d}\n' "$1" "$(date -Is)" $$ > "$CNC/chain.json"; }
fail() { echo "$1" > "$CNC/CHAIN-FAILED"; stage "failed: $1"; exit 1; }
rm -f "$CNC/CHAIN-FAILED"

[ -f cnf/h3p.cnf ] || $PY "$CODE"/cnfproof.py build --hidden 3 --bias general --require-used --out cnf/h3p
# first split: unit 0's input weights on cells 8..4 (243 cubes); timed-out cubes split 2 trits at a time
[ -f cnf/h3p-cubes.icnf ] || $PY "$CODE"/cnfproof.py cubes --cnf cnf/h3p.cnf --out cnf/h3p-cubes.icnf \
    --trits w1:0:8,w1:0:7,w1:0:6,w1:0:5,w1:0:4
stage conquer
$PY "$CODE"/cnfproof.py conquer --cnf cnf/h3p.cnf --cubes cnf/h3p-cubes.icnf --dir "$CNC" --workers "$WORKERS" \
    --cube-timeout "${CUBE_TIMEOUT:-1800}" --max-level "${MAX_LEVEL:-5}" --split trits --split-depth 2 \
    --final-timeout "${FINAL_TIMEOUT:-1209600}"
if [ -f "$CNC/DONE-SAT" ]; then touch "$CNC/CHAIN-H3-SAT"; stage "done: H=3 SAT (net in $CNC/nets)"; exit 0; fi
[ -f "$CNC/DONE-UNSAT" ] || fail "conquer did not finish UNSAT"

stage cover
$PY "$CODE"/cnfproof.py cover --cnf cnf/h3p.cnf --dir "$CNC" --timeout 172800 || fail "cover check"
touch "$CNC/CHAIN-H3-UNSAT"

# H=3 UNSAT (general) makes require-used sound at H=4. Best known at H=4: 39 nonzero (weights/smallest-never-loses-h4.json).
# Descend: nonzero <= K; SAT with n -> K = n - 1; UNSAT -> the minimum at H=4 is K + 1.
K=${H4_START:-38}
while [ "$K" -ge 0 ]; do
    stage "h4-nz$K"
    $PY "$CODE"/cnfproof.py build --hidden 4 --bias general --require-used --max-nonzero $K --out cnf/h4p-nz$K
    D=runs/h4-nz$K
    $PY "$CODE"/cnfproof.py portfolio --name h4-nz$K --dir $D --timeout 1209600 --keep-proof \
        --member "cnf/h4p-nz$K.cnf|cadical|0|1" --member "cnf/h4p-nz$K.cnf|kissat|0|0" \
        --member "cnf/h4p-nz$K.cnf|kissat|1|0|--sat"
    if [ -f $D/DONE-UNSAT ]; then stage "done: H=4 minimum is $((K + 1)) nonzero (proved)"; exit 0; fi
    [ -f $D/DONE-SAT ] || { stage "done: h4-nz$K undecided (see $D/status.json)"; exit 0; }
    N=$($PY -c "import json,sys; print(json.load(open('$D/DONE-SAT'))['nonzero'])")
    K=$((N - 1))
done
