#!/usr/bin/env bash
# Liane's method on the fire task, offline, first version (A99). Written on the laptop; NOT yet run on Della.
# Run on della-vis1 (the model calls need the internet), from inside the git clone:
#   cd ~/thinking-fast-hazard && bash scripts/liane_offline.sh ~/iw-thinking-fast/.secrets/anthropic_key.txt 2>&1 | tee logs/liane_offline.log
# The key is given as a file path only; this script never reads or prints it.
# Steps: (1) Haiku plays training rooms 100-109, asked before every command, fire routes on the menu;
#        (2-5) three rounds: Opus writes a program, it is audited, it plays validation rooms 110-119 alone;
#        (6) the round with the best validation mean plays rooms 50-99 once; (7) the costs are added up.
# Program runs use no model; they go to a compute node through srun (set RUN="" to run them here instead).
# It stops writing new rounds once the model spending passes LIMIT dollars (default 1.60, so the whole list stays under 2).
set -uo pipefail
KEY=${1:?give the path of the Anthropic key file}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT"
PY=${PY:-$ROOT/pyk310/bin/python}
RUN=${RUN-srun --ntasks=1 --cpus-per-task=1 --mem=4G --time=00:30:00 env PYTHONPATH=$ROOT}
OUT=${OUT:-results/firelab_compiled/v1}; TRAIN=${TRAIN:-haiku_train}; LIMIT=${LIMIT:-1.60}
spent() { "$PY" - "$OUT" "$TRAIN" <<'EOF'
import csv, glob, json, sys
out, train = sys.argv[1], sys.argv[2]
haiku = sum(float(r["cost_usd"]) for p in glob.glob(f"results/firelab_sim/{train}/rows_*.csv") for r in csv.DictReader(open(p)))
writer = sum(json.load(open(p)).get("usage", {}).get("dollars", 0.0) for p in glob.glob(f"{out}/compile_round*.json"))
print(f"{haiku + writer:.4f}")
EOF
}

echo "== [1] Haiku plays rooms 100-109"
"$PY" -m firelab.run_rooms --tag "$TRAIN" --first 100 --rooms 10 --agents model --model claude-haiku-5-5 --key-file "$KEY" \
    --call-time measured --no-board 2>&1 | grep -v "inertial\|b3Warning\|_link$" | tail -4

prev=""; val=""
for round in 1 2 3; do
  if "$PY" -c "import sys; sys.exit(0 if float('$(spent)') < float('$LIMIT') else 1)"; then :; else echo "== spending $(spent) dollars has passed $LIMIT: no more rounds"; break; fi
  echo "== [2] round $round: Opus writes a program"
  if "$PY" -m firelab.compile --games "results/firelab_sim/$TRAIN/games" --out "$OUT" --round "$round" --key-file "$KEY" \
        ${prev:+--previous "$prev"} ${val:+--validation "$val"} | tail -12; then
    echo "== [4] round $round: the program plays rooms 110-119"
    $RUN "$PY" -m firelab.validate --program "$OUT/policy_round$round.py" --first 110 --rooms 10 --out "$OUT/validation_round$round.json" 2>&1 \
        | grep -v "inertial\|b3Warning\|_link$" | tail -16
    [ -f "$OUT/validation_round$round.json" ] && prev="$OUT/policy_round$round.py" && val="$OUT/validation_round$round.json"
  else
    echo "== round $round gave no accepted program (see $OUT/compile_round$round.json)"
  fi
done

best=$("$PY" -c "import glob, json; files = glob.glob('$OUT/validation_round*.json'); print(max(files, key=lambda p: json.load(open(p))['mean']) if files else '')")
[ -n "$best" ] || { echo "== no round produced a program that played; nothing to test"; exit 1; }
round=${best##*validation_round}; round=${round%.json}
echo "== [6] best validation round: $round; it plays rooms 50-99 once"
$RUN "$PY" -m firelab.validate --program "$OUT/policy_round$round.py" --first 50 --rooms 50 --out "$OUT/test_rooms_50_99.json" 2>&1 \
    | grep -v "inertial\|b3Warning\|_link$" | tail -16

echo "== [7] preparation cost"
"$PY" - "$OUT" "$TRAIN" <<'EOF'
import csv, glob, json, sys
out, train = sys.argv[1], sys.argv[2]
rows = [r for p in glob.glob(f"results/firelab_sim/{train}/rows_*.csv") for r in csv.DictReader(open(p))]
print(f"Haiku games: {len(rows)} rooms, mean score {sum(float(r['sim']) for r in rows) / max(1, len(rows)):.3f}, "
      f"{sum(float(r['measured_s']) for r in rows):.0f} s of model time, {sum(float(r['cost_usd']) for r in rows):.4f} dollars")
for p in sorted(glob.glob(f"{out}/compile_round*.json")):
    c = json.load(open(p)); u = c.get("usage", {})
    print(f"writer round {c['round']}: accepted {c.get('accepted')}, {c.get('seconds')} s, tokens in {u.get('input_tokens')} out {u.get('output_tokens')}, {u.get('dollars')} dollars")
for p in sorted(glob.glob(f"{out}/validation_round*.json")) + glob.glob(f"{out}/test_rooms_50_99.json"):
    v = json.load(open(p))
    print(f"{p}: mean {v['mean']} (reflex {v.get('reflex_mean')}), gap {v.get('gap_to_reflex')} {v.get('gap_interval')}, deaths {v['deaths']}, illegal {v['illegal']}, "
          f"raised {v['raised']}, {1000 * v['seconds_per_decision']:.2f} ms per decision")
EOF
echo "== done; total model spending $(spent) dollars. Compare the test line with rooms 50-99: reflex 0.627, Jev 0.676, planner 0.888, cautious planner 0.932."
