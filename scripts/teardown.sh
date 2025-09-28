#!/usr/bin/env bash
set -euo pipefail

PGID_FILE="/tmp/rlvo_bringup.pgid"
if [[ ! -f "${PGID_FILE}" ]]; then
  echo "No PGID file found at ${PGID_FILE}"
  exit 0
fi

PGID=$(cat "${PGID_FILE}")
if [[ -n "${PGID}" ]]; then
  echo "Killing process group PGID=${PGID}"
  kill -TERM "-${PGID}" || true
  sleep 2
  # force kill if still alive
  kill -KILL "-${PGID}" || true
else
  echo "Empty PGID file"
fi

rm -f "${PGID_FILE}"
