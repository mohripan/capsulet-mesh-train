#!/bin/sh
set -eu

# Only resources created by this invocation are removed by the exit trap.
image=${1:-meshtrain-native-test:dev}
suffix="$(date +%s)-$$"
network="meshtrain-tcp-$suffix"
root="meshtrain-root-$suffix"
client="meshtrain-client-$suffix"
worker=/workspace/build/native/tests/meshtrain_tcp_tests
cleanup() {
    docker rm -f "$client" "$root" >/dev/null 2>&1 || true
    docker network rm "$network" >/dev/null 2>&1 || true
}
trap cleanup EXIT HUP INT TERM

docker network create "$network" >/dev/null
docker run --detach --name "$root" --network "$network" --network-alias meshtrain-root \
    --entrypoint "$worker" "$image" --worker container 0 2 45000 meshtrain-root >/dev/null
docker run --name "$client" --network "$network" --entrypoint "$worker" \
    "$image" --worker container 1 2 45000 meshtrain-root
status=$(docker wait "$root")
docker logs "$root"
test "$status" = 0
echo 'PASS two-container hostname TCP exchange (524295 bytes each way)'
