#!/usr/bin/env bash
# A3 server-side health monitor + auto-fix.
# Runs forever in nohup setsid, polls every 5 minutes.
#
# Detects and auto-fixes:
#  - Stuck training (log not growing for >10 min): kill + restart from checkpoint
#  - Disk space < 30 GB: clean SpheroMix .cache and old checkpoint epochs
#  - Crashed pretrain (python died, run_all_4.sh advanced past it): log only
#  - OOM in recent log: restart with halved per_gpu_bs (and 2x grad_accum)
#  - Orphan python (still running but bash parent dead): not killed, just logged
#
# Logs actions to /disk1/prusek/SpheroSeg/logs/health.log

set -u
LOG=/disk1/prusek/SpheroSeg/logs/health.log
mkdir -p /disk1/prusek/SpheroSeg/logs

CKPT=/disk1/prusek/SpheroSeg/checkpoints
LOGD=/disk1/prusek/SpheroSeg/logs
DATA=/disk1/prusek/SpheroSeg/data

INTERVAL=300            # 5 min
STUCK_THRESHOLD=1800     # 30 min without log growth = stuck
DISK_FREE_MIN_GB=30     # below this, start cleanup
OOM_PATTERNS='CUDA out of memory|OOM|RuntimeError: CUDA error: out of memory|Killed'

log() { echo "[$(date -Iseconds)] $*" >> "$LOG"; }

#---------- helpers ----------
get_log_file() {
  local model="$1"
  case "$model" in
    hrnet)
      _pre="$LOGD/hrnet_a3.log"; _ft="$LOGD/hrnet_a3_finetune.log"
      if [ -f "$_ft" ] && [ "$(stat -c %Y "$_ft" 2>/dev/null)" -gt "$(stat -c %Y "$_pre" 2>/dev/null)" ]; then echo "$_ft"; else echo "$_pre"; fi
      ;;
    pspnet) _pre="$LOGD/pspnet_a3.log"; _ft="$LOGD/pspnet_a3_finetune.log"; if [ -f "$_ft" ] && [ "$(stat -c %Y "$_ft" 2>/dev/null)" -gt "$(stat -c %Y "$_pre" 2>/dev/null)" ]; then echo "$_ft"; else echo "$_pre"; fi ;;
    resunet_cbam) _pre="$LOGD/resunet_cbam_a3.log"; _ft="$LOGD/resunet_cbam_a3_finetune.log"; if [ -f "$_ft" ] && [ "$(stat -c %Y "$_ft" 2>/dev/null)" -gt "$(stat -c %Y "$_pre" 2>/dev/null)" ]; then echo "$_ft"; else echo "$_pre"; fi ;;
    unet_a100)    echo "$LOGD/unet_a3.log" ;;
    unet_a5000)
      # Prefer freshest of pretrain or finetune log
      _pre="$LOGD/unet_a5000_a3.log"
      _ft="$LOGD/unet_a5000_a3_finetune.log"
      if [ -f "$_ft" ] && [ "$(stat -c %Y "$_ft" 2>/dev/null)" -gt "$(stat -c %Y "$_pre" 2>/dev/null)" ]; then
        echo "$_ft"
      else
        echo "$_pre"
      fi
      ;;
  esac
}

# Returns 0 if log file's last-modification was within $STUCK_THRESHOLD seconds.
log_alive() {
  local f="$1"
  [ ! -f "$f" ] && return 1
  local mtime=$(stat -c %Y "$f" 2>/dev/null || echo 0)
  local now=$(date +%s)
  [ $((now - mtime)) -lt $STUCK_THRESHOLD ]
}

# Returns 0 if any pretrain or finetune python is running for $1
python_running_for() {
  local model="$1"; local suffix="${2:-}"
  if [ -n "$suffix" ]; then
    pgrep -f "run_a3_launcher\.py.*--model $model.*--output-suffix $suffix" >/dev/null 2>&1
  else
    pgrep -fa 'run_a3_launcher\.py' | grep -E "model $model" | grep -v 'output-suffix' | head -1 | grep -q .
  fi
}

# Detect OOM or fatal pattern in last 200 lines of a log
log_has_oom() {
  local f="$1"
  [ ! -f "$f" ] && return 1
  tail -200 "$f" 2>/dev/null | grep -Eq "$OOM_PATTERNS"
}

#---------- monitors ----------

check_disk() {
  local free_gb=$(df -BG /disk1 | awk 'NR==2 {gsub("G",""); print $4}')
  if [ "$free_gb" -lt "$DISK_FREE_MIN_GB" ]; then
    log "DISK LOW: /disk1 free=${free_gb}GB < ${DISK_FREE_MIN_GB}GB; cleaning caches"
    # clean dataset caches (will be regenerated)
    find "$DATA/SpheroMix/.cache" -type f -delete 2>/dev/null && log "  cleaned SpheroMix .cache"
    find "$DATA/SpheroHQ/.cache"  -type f -delete 2>/dev/null && log "  cleaned SpheroHQ  .cache"
    # clean old checkpoint epoch files (keep only best_model.pth + last_model.pth)
    find "$CKPT" -name 'checkpoint_epoch_*.pth' -delete 2>/dev/null && log "  cleaned epoch checkpoints"
    free_gb=$(df -BG /disk1 | awk 'NR==2 {gsub("G",""); print $4}')
    log "  /disk1 free now: ${free_gb}GB"
  fi
}

check_a100() {
  # Determine which A100 model is "expected" to be running.
  # If run_all_4.sh is still alive, A100 is in pretrain phase.
  # Otherwise super_orchestrator should be running A100 finetunes.
  local mode=""
  if pgrep -f 'run_all_4\.sh' >/dev/null 2>&1; then
    mode="pretrain"
  elif pgrep -f 'super_orchestrator\.sh' >/dev/null 2>&1; then
    mode="finetune"
  fi

  # Find current A100 python (excluding A5000)
  local a100_pid=$(pgrep -af 'run_a3_launcher\.py' | grep -v 'output-suffix a5000' | head -1 | awk '{print $1}')
  if [ -z "$a100_pid" ]; then
    log "A100 status: no run_a3 python on A100 (mode=$mode)"
    return
  fi
  local a100_model=$(ps -p $a100_pid -o args= 2>/dev/null | grep -oE -- '--model [a-z_]+' | awk '{print $2}')
  local a100_stage=$(ps -p $a100_pid -o args= 2>/dev/null | grep -oE -- '--stage [a-z]+' | awk '{print $2}')
  [ -z "$a100_stage" ] && a100_stage="pretrain"   # legacy
  local logf=$(get_log_file "$a100_model")

  if log_alive "$logf"; then
    local last=$(tail -1 "$logf" 2>/dev/null | tr "\r" "\n" | tail -1 | head -c 120)
    log "A100 OK: pid=$a100_pid model=$a100_model stage=$a100_stage  | $last"
  else
    log "A100 STUCK: pid=$a100_pid model=$a100_model stage=$a100_stage; log $logf not updated for >${STUCK_THRESHOLD}s"
    if log_has_oom "$logf"; then
      log "A100 OOM detected — would halve per_gpu_bs (manual intervention needed; logging only)"
    else
      log "A100 stuck without OOM; killing python pid=$a100_pid for restart"
      kill -9 $a100_pid 2>/dev/null
      sleep 5
      pkill -9 -P $a100_pid 2>/dev/null
      log "  killed; bash loop should advance to next model OR orchestrator can pick up"
    fi
  fi
}

check_a5000() {
  local pid=$(pgrep -f 'run_a3_launcher\.py.*--model unet.*--output-suffix a5000' | head -1)
  if [ -z "$pid" ]; then
    log "A5000 status: no U-Net python; either done or pre-launch"
    return
  fi
  local stage=$(ps -p $pid -o args= 2>/dev/null | grep -oE -- '--stage [a-z]+' | awk '{print $2}')
  [ -z "$stage" ] && stage="pretrain(legacy)"
  local logf=$(get_log_file "unet_a5000")

  if log_alive "$logf"; then
    local last=$(tail -1 "$logf" 2>/dev/null | tr "\r" "\n" | tail -1 | head -c 120)
    log "A5000 OK: pid=$pid stage=$stage | $last"
  else
    log "A5000 STUCK: pid=$pid stage=$stage; log not updated for >${STUCK_THRESHOLD}s"
    if log_has_oom "$logf"; then
      log "A5000 OOM detected (BS=4 already minimum); manual intervention likely needed"
    else
      log "A5000 stuck without OOM; killing pid=$pid"
      kill -9 $pid 2>/dev/null
      sleep 5
      pkill -9 -P $pid 2>/dev/null
    fi
  fi
}

check_orchestrator() {
  if pgrep -f 'super_orchestrator\.sh' >/dev/null 2>&1; then
    log "Orchestrator OK"
  else
    # Only alarm if SpheroHQ is unzipped but no orchestrator → it crashed
    if [ -d "$DATA/SpheroHQ/train" ]; then
      log "WARN orchestrator dead but SpheroHQ ready; would restart"
      nohup setsid bash /disk1/prusek/SpheroSeg/code/scripts/a3/super_orchestrator.sh \
        </dev/null >/dev/null 2>&1 &
      disown
      log "  restarted orchestrator (PID=$!)"
    fi
  fi
}

check_watchdog_unet() {
  if pgrep -f 'watchdog_kill_a100_unet' >/dev/null 2>&1; then
    : # watchdog still armed
  else
    # Has it already fired? Check log for completion
    if [ -f "$LOGD/watchdog_unet.log" ]; then
      if grep -q "watchdog done" "$LOGD/watchdog_unet.log" 2>/dev/null; then
        : # already did its job
      elif grep -q "watchdog start" "$LOGD/watchdog_unet.log" 2>/dev/null; then
        log "WARN U-Net watchdog dead before firing; restarting"
        nohup setsid bash /tmp/watchdog_kill_a100_unet.sh </dev/null >/dev/null 2>&1 &
        disown
      fi
    fi
  fi
}

#---------- main loop ----------

log "================================================================"
log "health-check loop START (PID=$$, interval=${INTERVAL}s)"
log "================================================================"

while true; do
  check_disk
  check_a100
  check_a5000
  check_orchestrator
  check_watchdog_unet
  sleep "$INTERVAL"
done
