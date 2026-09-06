#!/usr/bin/env bash
# Keep a diffusion trainer alive until a wall-clock deadline.
#
# A headless renderer running WebGPU for hours occasionally has its context
# destroyed — twice now, once after a long stall under GPU contention. The
# trainer itself is fine with that: it writes a resumable checkpoint every
# three minutes, so a crash costs a few minutes rather than the run. What it
# cannot do is restart itself. This loop does, in chunks, always with --resume.
#
# Usage: scripts/train-long.sh <objective> <total-hours> [extra args…]
set -u

OBJECTIVE="$1"
HOURS="$2"
shift 2

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
END=$(( $(date +%s) + $(python3 -c "print(int($HOURS*3600))") ))
LOG=".cache/logs/long-$OBJECTIVE.log"
CHUNK_MAX=60
ATTEMPT=0

while :; do
	NOW=$(date +%s)
	LEFT_MIN=$(( (END - NOW) / 60 ))
	if [ "$LEFT_MIN" -lt 5 ]; then
		echo "[supervisor] $OBJECTIVE: deadline reached" | tee -a "$LOG"
		break
	fi
	CHUNK=$LEFT_MIN
	[ "$CHUNK" -gt "$CHUNK_MAX" ] && CHUNK=$CHUNK_MAX
	ATTEMPT=$(( ATTEMPT + 1 ))
	echo "[supervisor] $OBJECTIVE: chunk $ATTEMPT, $CHUNK min, $LEFT_MIN min left" | tee -a "$LOG"
	node scripts/train-emoji.mjs --objective "$OBJECTIVE" --minutes "$CHUNK" --resume "$@" >>"$LOG" 2>&1
	echo "[supervisor] $OBJECTIVE: chunk $ATTEMPT exited $?" | tee -a "$LOG"
	# let the GPU settle before the next renderer asks for a device
	sleep 20
done
