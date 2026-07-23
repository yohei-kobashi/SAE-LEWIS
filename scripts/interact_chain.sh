#!/bin/bash
# Generic interact-g chain driver v3 (2026-07-23).
# v2 fix kept: the command is written to the session's stdin only AFTER
# "job ... ready" appears (PBS flushes early input), and stdin stays open
# for the whole session.
# v3 fix (user 2026-07-23): the stdin holder used to be the left side of a
# pipeline, so after a walltime kill the driver waited for its sleep to
# expire (up to ~8 min lost per session). Now the holder writes into a
# FIFO in the background and is KILLED the moment ssh exits — the next
# session launches immediately.
#
# usage: interact_chain.sh <name> <remote_cmd> <done_marker> <remote_done_test> [max_real]
#   name             log prefix (logs: $CHAIN_LOGDIR/<name>_e<i>.log)
#   remote_cmd       command typed into the session (should end with '; exit')
#   done_marker      string in the session log that means the chain is done
#   remote_done_test remote shell test, e.g. 'test -f /path/report.md'
#   max_real         real-session budget (default 8)
# env: CHAIN_LOGDIR (default /tmp), MSSH (default: ssh miyabi/miyabi-c failover)

NAME=$1; CMD=$2; MARKER=$3; DONE_TEST=$4; MAXREAL=${5:-8}
LOGD=${CHAIN_LOGDIR:-/tmp}
MSSH_BIN=${MSSH:-}
msshc () {
  if [ -n "$MSSH_BIN" ]; then $MSSH_BIN "$@"; return; fi
  local TT=""
  if [ "$1" = "-tt" ]; then TT="-tt"; shift; fi
  for H in miyabi miyabi-c; do
    if timeout 20 ssh -o ConnectTimeout=12 -o BatchMode=yes "$H" true 2>/dev/null; then
      ssh $TT "$H" "$@"; return
    fi
  done
  echo "MSSH-NO-HOST-REACHABLE" >&2
  return 63
}

real=0
for i in $(seq 1 80); do
  timeout 60 bash -c "$(declare -f msshc); msshc '$DONE_TEST'" 2>/dev/null
  if [ $? -eq 0 ]; then
    echo "CHAIN-COMPLETE (remote done-test; attempt $i, real $real)"; break
  fi
  A=$LOGD/${NAME}_e$i.log
  : > "$A"
  FIFO=$(mktemp -u "$LOGD/${NAME}_fifo.XXXX")
  mkfifo "$FIFO"
  (
    t=0
    until grep -q "ready" "$A" 2>/dev/null; do
      sleep 5; t=$((t+5))
      if [ $t -ge 900 ]; then exit 0; fi
    done
    sleep 15
    printf '%s\n' "$CMD"
    sleep "${CHAIN_HOLD:-8000}"   # > walltime; killed as soon as ssh exits
  ) > "$FIFO" &
  W=$!
  msshc -tt "qsub -I -l select=1 -W group_list=go25 -q ${CHAIN_QUEUE:-interact-g}" \
      < "$FIFO" > "$A" 2>&1
  kill "$W" 2>/dev/null; wait "$W" 2>/dev/null
  rm -f "$FIFO"
  if grep -q "$MARKER" "$A"; then
    echo "CHAIN-COMPLETE (attempt $i, real $real)"; break; fi
  if grep -q "would exceed per-user limit\|qsub: Error\|Permission denied\|MSSH-NO-HOST-REACHABLE" "$A"; then
    sleep 45; continue; fi
  if ! grep -qF "${CMD%%;*}" "$A"; then
    echo "ANOMALY: session $i ended with no command echo (see $A)"
  fi
  real=$((real+1))
  if [ $real -ge "$MAXREAL" ]; then echo "REAL-SESSION-BUDGET-EXHAUSTED"; break; fi
  sleep 20
done
echo "DRIVER-ENDED real=$real"
tail -8 "$A" 2>/dev/null | tr -d "\r" | grep -v "^$" | tail -5
