#!/bin/bash
# Run ON THE CLUSTER in the repo root, after the first `bash tools/sync_to_cluster.sh`.
# Moves the cluster's own files from the old flat layout into the new one.
#   bash tools/migrate_cluster.sh            dry run: prints the plan, changes nothing
#   bash tools/migrate_cluster.sh --apply    does it
# Never deletes or overwrites: a move happens only if the source exists and the destination
# does not; otherwise "skip (exists): <dest>" is printed. Safe to run again.
set -euo pipefail
shopt -s nullglob dotglob

APPLY=0
case "${1:-}" in
    "") ;;
    --apply) APPLY=1 ;;
    *) echo "usage: bash tools/migrate_cluster.sh [--apply]" >&2; exit 1 ;;
esac
if [ ! -f tools/layout.py ]; then
    echo "error: run this from the repo root (tools/layout.py not found in $PWD)" >&2
    exit 1
fi

# act VERB SRC DST: mv/cp SRC to DST only if SRC exists and DST does not. Returns 0 if it acted.
act() {
    local verb=$1 src=$2 dst=$3
    { [ -e "$src" ] || [ -L "$src" ]; } || return 1
    if [ -e "$dst" ] || [ -L "$dst" ]; then
        echo "skip (exists): $dst"
        return 1
    fi
    echo "$verb $src $dst"
    if [ "$APPLY" = 1 ]; then
        mkdir -p "$(dirname "$dst")"
        "$verb" -n "$src" "$dst"
    fi
}

# Old root files are archived (the sync already put the new copies in jobs/, containers/, data_tools/).
act mv code archive/code/cluster || true
for f in job_radfm.sh Dockerfile prezemi_podatoci.sh proveri_podatoci.py a.py; do
    act mv "$f" "archive/cluster_root/$f" || true
done
for f in prezemi_molab.py sporedi_molab.py linkovi.txt molab_373_datoteki.txt *.log; do
    [ -f "$f" ] && { act mv "$f" "data_tools/$f" || true; }
done

# results/: slurm logs, then everything else at the top level into a dated archive folder.
for f in results/*.out; do
    [ -f "$f" ] && { act mv "$f" "results/slurm/$(basename "$f")" || true; }
done
archive="results/archive/cluster_$(date +%F)"
copies=()
for f in results/*; do
    [ -f "$f" ] || continue
    case "$f" in *.out) continue ;; esac
    if act mv "$f" "$archive/$(basename "$f")"; then
        case "$(basename "$f")" in
            radfm_*_processed.jsonl|radfm_*_details.jsonl) copies+=("$(basename "$f")") ;;
        esac
    fi
done
# Only files moved in this run are copied, so a second run finds nothing to do.
for name in ${copies[@]+"${copies[@]}"}; do
    act cp "$archive/$name" "results/radfm/full/$name" || true
done

act mv RadFM third_party/RadFM || true
if [ -d pyuser ] && [ ! -e pyuser/radfm ]; then
    for f in pyuser/*; do
        act mv "$f" "pyuser/radfm/$(basename "$f")" || true
    done
fi

if [ "$APPLY" = 0 ]; then
    echo "dry run: nothing changed; run with --apply to do it"
fi

# Anything else at the root is not ours: report it, never touch it.
known=" README.md splits.json .gitignore .gitattributes .manu_commit .git .venv .idea .pytest_cache .superpowers
 __pycache__ common models jobs containers data_tools evaluation tools reference results archive tests docs
 data hf_cache pyuser third_party code RadFM job_radfm.sh Dockerfile prezemi_podatoci.sh proveri_podatoci.py a.py
 prezemi_molab.py sporedi_molab.py linkovi.txt molab_373_datoteki.txt "
known=${known//$'\n'/ }
for f in *; do
    case "$known" in *" $f "*) continue ;; esac
    case "$f" in *.log) continue ;; esac
    echo "left in place: $f"
done

[ "$APPLY" = 1 ] || exit 0
mkdir -p results/slurm results/radfm/full third_party pyuser hf_cache containers
python3 tools/layout.py --cluster .
