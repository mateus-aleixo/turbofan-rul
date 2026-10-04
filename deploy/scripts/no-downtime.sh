#!/usr/bin/env bash
# Assert that a disruption costs zero requests.
#
#   no-downtime.sh rollout    roll every pod while traffic is flowing
#   no-downtime.sh kill-pod   delete one pod while traffic is flowing
#
# A poller hammers the NodePort throughout and counts every response that is not
# 200. The claim being tested is the pair maxUnavailable=0 plus the preStop
# drain: without either one, this script fails, which is the point of having it.
set -euo pipefail

MODE="${1:-rollout}"
RELEASE="${RELEASE:-rul}"
ENDPOINT="${ENDPOINT:-http://localhost:30080}"
DEPLOY="deploy/${RELEASE}-conformal-rul"
FAILS=$(mktemp)
TOTAL=$(mktemp)
STOP=$(mktemp -u)

poll() {
  while [ ! -f "$STOP" ]; do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "$ENDPOINT/health" || echo 000)
    echo x >> "$TOTAL"
    [ "$code" = "200" ] || echo "$code" >> "$FAILS"
    sleep 0.05
  done
}

poll &
POLLER=$!
# Let the poller establish a baseline before anything is disrupted, so a failure
# cannot be blamed on traffic starting mid-disruption.
sleep 3
baseline=$(wc -l < "$TOTAL")
# The bar is deliberately low. Probe rate is a property of the machine running
# this (one curl process per probe, which is far slower on Windows than on a CI
# runner), not of the service. What matters is that traffic is flowing and that
# none of it is failing before the disruption starts.
if [ "$baseline" -lt 5 ]; then
  echo "::error::poller only made $baseline requests in 3s, the endpoint is not answering"
  touch "$STOP"; wait "$POLLER" || true; exit 1
fi
if [ -s "$FAILS" ]; then
  echo "::error::endpoint was already failing before the disruption"
  sort "$FAILS" | uniq -c
  touch "$STOP"; wait "$POLLER" || true; exit 1
fi

before=$(kubectl get pods -l "app.kubernetes.io/instance=$RELEASE" \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' | sort | tr '\n' ' ')

case "$MODE" in
  rollout)
    kubectl rollout restart "$DEPLOY"
    kubectl rollout status "$DEPLOY" --timeout=3m
    ;;
  kill-pod)
    victim=$(kubectl get pods -l "app.kubernetes.io/instance=$RELEASE" \
      -o jsonpath='{.items[0].metadata.name}')
    echo "deleting $victim"
    kubectl delete pod "$victim" --wait=true
    kubectl rollout status "$DEPLOY" --timeout=3m
    ;;
  *)
    echo "unknown mode: $MODE" >&2; touch "$STOP"; wait "$POLLER" || true; exit 2
    ;;
esac

# Keep polling briefly after the disruption reports success: the interesting
# failures are the ones that land just after a pod is declared Ready.
sleep 3
touch "$STOP"
wait "$POLLER" || true

after=$(kubectl get pods -l "app.kubernetes.io/instance=$RELEASE" \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' | sort | tr '\n' ' ')
total=$(wc -l < "$TOTAL")
fails=$(wc -l < "$FAILS" || echo 0)

echo "mode:      $MODE"
echo "pods before: $before"
echo "pods after:  $after"
echo "requests:  $total"
echo "failures:  $fails"

if [ "$before" = "$after" ]; then
  echo "::error::pod set did not change, so nothing was actually disrupted"
  exit 1
fi
if [ "$fails" -ne 0 ]; then
  echo "::error::$fails of $total requests failed during $MODE"
  sort "$FAILS" | uniq -c
  exit 1
fi
# Zero failures out of five probes would prove nothing. This is the guard that
# keeps a green result meaningful rather than merely green.
if [ "$total" -lt 20 ]; then
  echo "::error::only $total probes covered the $MODE window, too few to claim anything"
  exit 1
fi
echo "$MODE completed with zero failed requests across $total probes"
