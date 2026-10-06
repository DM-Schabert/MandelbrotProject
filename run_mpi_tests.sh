#!/usr/bin/env bash
# Run the MPI tests under 1, 2, 3 and 4 ranks.
#
# --timeout-method=thread: the default (signal) can't interrupt a rank that is
# stuck inside an MPI call, so a hang would never time out. The thread method
# can, and it ends the stuck rank so mpirun stops the rest.
# The outer `timeout` is a last safety net for the whole run.
#
# Open MPI refuses to start more ranks than the machine has cores unless it
# gets --oversubscribe. MPICH has no such limit (and doesn't know the flag),
# so the flag is only added for Open MPI.

set -u
cd "$(dirname "$0")"   # run from the project folder, wherever it's called from

TEST_FILE=tests/test_collect.py
if [ ! -f "$TEST_FILE" ]; then
    echo "Can't find $TEST_FILE (looked in $(pwd))"
    exit 1
fi

# Ask mpi4py which MPI library it uses; that's more reliable than parsing
# `mpirun --version`, whose output differs between builds.
VENDOR=$(python3 -c "from mpi4py import MPI; print(MPI.get_vendor()[0])" 2>/dev/null)
MPI_FLAGS=""
if [ "$VENDOR" = "Open MPI" ]; then
    MPI_FLAGS="--oversubscribe"
fi
echo "MPI library: ${VENDOR:-unknown}   extra mpirun flags: ${MPI_FLAGS:-none}"

status=0
for n in 1 2 3 4; do
    echo "=== $n rank(s) ==="
    if ! timeout 300 mpirun $MPI_FLAGS -n "$n" python3 -m pytest --with-mpi -q \
            --timeout=60 --timeout-method=thread -p no:cacheprovider \
            "$TEST_FILE"; then
        echo "*** FAILED with $n rank(s) ***"
        status=1
    fi
done

if [ $status -eq 0 ]; then
    echo "All MPI tests passed with 1-4 ranks."
fi
exit $status