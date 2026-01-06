#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/va/projects/sprint7/RAG"
PY="$ROOT/.venv/bin/python"

LOG="$ROOT/logs/update.log"
mkdir -p "$ROOT/logs"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

log "START update"

log "1) chunk_documents_langchain.py"
$PY "$ROOT/scripts/chunk_documents_langchain.py" >>"$LOG" 2>&1

log "2) embed_chunks.py"
$PY "$ROOT/scripts/embed_chunks.py" >>"$LOG" 2>&1

log "3) build_faiss_index.py"
$PY "$ROOT/scripts/build_faiss_index.py" >>"$LOG" 2>&1

log "DONE update"
