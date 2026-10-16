#!/bin/sh
set -eu

image=${1:-meshtrain-native-test:dev}
prefix="meshtrain-collective-$(date +%s)-$$"
worker=/workspace/build/native/tests/meshtrain_collective_tests
cleanup() {
    docker rm -f "$prefix-0" "$prefix-1" "$prefix-2" "$prefix-3" >/dev/null 2>&1 || true
    docker network rm "$prefix" >/dev/null 2>&1 || true
}
trap cleanup EXIT HUP INT TERM

docker network create "$prefix" >/dev/null
for rank in 0 1 2 3; do
    docker run --detach --name "$prefix-$rank" --network "$prefix" \
        --network-alias "meshtrain-worker-$rank" --entrypoint "$worker" "$image" \
        --worker container "$rank" 4 45000 meshtrain-worker-0 >/dev/null
done

failed=0
for rank in 0 1 2 3; do
    status=$(docker wait "$prefix-$rank")
    docker logs "$prefix-$rank"
    if test "$status" != 0; then failed=1; fi
done
test "$failed" = 0
echo 'PASS four-container centralized SUM: 14 collectives per rank, exact + random fixtures'
