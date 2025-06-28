#!/usr/bin/env bash
# File: scripts/performance_regression.sh
# Purpose: Latency Spike Simulation against local Chatbot HTTP endpoint

set -eo pipefail

# Configuration (override via environment variables)
REDIS_IMAGE="${REDIS_IMAGE:-redis:7-alpine}"
REDIS_PORT="${REDIS_PORT:-6379}"
CHATBOT_URL="${CHATBOT_URL:-http://127.0.0.1:8000/v1/chat/completions}"
DURATION="${DURATION:-30s}"
QPS="${QPS:-40}"
BASELINE_FILE="tests/performance/baseline.json"
MAX_P99_RATIO="${MAX_P99_RATIO:-1.15}"

# Start temporary Redis
sudo docker run -d --rm --name perf-redis -p "${REDIS_PORT}":6379 "${REDIS_IMAGE}"

# Introduce network latency and jitter on loopback
sudo tc qdisc add dev lo root netem delay 180ms 40ms distribution normal

# Skip I/O throttle test if blkio directory not found
if [ -e /sys/fs/cgroup/blkio/blkio.throttle.write_bps_device ]; then
  # Throttle disk write throughput (cgroup v1)
  echo "8:0 ${WRITE_BPS:-1048576}" | sudo tee /sys/fs/cgroup/blkio/blkio.throttle.write_bps_device >/dev/null
fi

# Run load test with hey and output JSON
echo "Running hey for ${DURATION} at ${QPS} QPS..."
hey -z "${DURATION}" -q "${QPS}" -o json "${CHATBOT_URL}" > current_stats.json

# Clean up network and I/O shaping
sudo tc qdisc del dev lo root || true

# Stop Redis
sudo docker stop perf-redis >/dev/null

# Extract P99 latency
CURRENT_P99=$(jq '.Latencies["99"]' current_stats.json)
echo "Current P99: ${CURRENT_P99} ms"

# If no baseline exists, save and exit success
if [ ! -f "${BASELINE_FILE}" ]; then
  mkdir -p "$(dirname "${BASELINE_FILE}")"
  jq '{p95: .Latencies["95"], p99: .Latencies["99"]}' current_stats.json \
    > "${BASELINE_FILE}"
  echo "Baseline saved to ${BASELINE_FILE}"
  exit 0
fi

# Read baseline P99
BASE_P99=$(jq '.p99' "${BASELINE_FILE}")
echo "Baseline P99: ${BASE_P99} ms"

# Compute threshold
THRESHOLD=$(awk "BEGIN {printf \"%f\", ${BASE_P99} * ${MAX_P99_RATIO}}")
echo "Threshold P99: ${THRESHOLD} ms"

# Compare and exit accordingly
if (( $(awk "BEGIN {print (${CURRENT_P99} > ${THRESHOLD})}") )); then
  echo "🚨 P99 regression detected! (${CURRENT_P99} > ${THRESHOLD})"
  exit 1
else
  echo "✅ Performance within acceptable range."
  exit 0
fi
