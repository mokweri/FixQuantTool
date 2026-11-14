#!/usr/bin/env bash
# Submit every run of the gate-6 matrix (DeepTile sparsity line) that has no
# result yet. The matrix and its recipes are fixed by the gate-6 protocol;
# do not edit them to chase a result.
#
#   scripts/nm_gate6_submit.sh            # list what would be submitted
#   scripts/nm_gate6_submit.sh --submit   # submit it
#
# A run counts as done when its run_manifest.yaml exists, and as pending when
# a job of the same name is queued or running; both are skipped. Running the
# script again after a crash therefore resubmits exactly the failed runs.
set -euo pipefail

SUBMIT=0
[[ "${1:-}" == "--submit" ]] && SUBMIT=1
STORAGE_BASE="${FIXQUANT_STORAGE_BASE:-/nobackup/proj/disk/naiss2024-22-1352/personal/mogaka}"
RESULTS="$STORAGE_BASE/results/qat-nm"
HERE="$(cd "$(dirname "$0")" && pwd)"
LISTS="$HERE/../configs/nm_layers"
SEEDS=(0 1 2)
PATTERNS=(dense oc_n16 pair24+perm filter24)

QUEUED="$(squeue -u "$USER" -h -o '%j' 2>/dev/null || true)"
n_new=0 n_done=0 n_queued=0

# one run: MODEL PATTERN FT_LR QAT_EPOCHS SEED [LIST]
run() {
    local model=$1 pattern=$2 ftlr=$3 epochs=$4 seed=$5 list=${6:-}
    local sel="" ; [[ -z "$list" ]] || sel="-sel_$list"
    local tag="${pattern/+/_}${sel}-ft${ftlr}x15-lr1e-5-e${epochs}-s${seed}"
    local name="nm-${model}-${pattern/+/_}${sel}-ft${ftlr}x15-s${seed}"
    if compgen -G "$RESULTS/$model/$tag/*/$model/run_manifest.yaml" >/dev/null; then
        n_done=$((n_done + 1)); return
    fi
    if grep -qxF "$name" <<<"$QUEUED"; then
        n_queued=$((n_queued + 1)); return
    fi
    n_new=$((n_new + 1))
    printf 'new  %s\n' "$tag ($model)"
    if ((SUBMIT)); then
        local args=("$model" "$pattern" "$epochs" 1e-5 "$seed")
        [[ -z "$list" ]] || args+=("$LISTS/$list.json")
        NM_FT_EPOCHS=15 NM_FT_LR="$ftlr" "$HERE/nm_submit.sh" "${args[@]}"
    fi
}

for seed in "${SEEDS[@]}"; do
    # ResNets: uniform 2:4 (every layer). ResNet-50 at two float lrs; the
    # protocol picks one from the dense runs alone.
    for p in "${PATTERNS[@]}"; do
        run resnet18 "$p" 3e-3 5 "$seed"
        for lr in 1e-3 3e-4; do run resnet50 "$p" "$lr" 5 "$seed"; done
    done
    # Mobile networks: uniform and the two selective cuts per pattern.
    for net in mobilenet_v2 efficientnet_lite0; do
        [[ $net == mobilenet_v2 ]] && E=10 || E=5
        for p in "${PATTERNS[@]}"; do
            run "$net" "$p" 1e-3 "$E" "$seed"
            [[ $p == dense ]] && continue
            for cut in s50 s75; do
                run "$net" "$p" 1e-3 "$E" "$seed" "${net}_${p/+/_}_${cut}"
            done
        done
    done
done

printf '\n%d done, %d queued or running, %d %s\n' "$n_done" "$n_queued" "$n_new" \
    "$( ((SUBMIT)) && echo submitted || echo 'to submit (rerun with --submit)')"
