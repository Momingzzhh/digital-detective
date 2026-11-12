#!/bin/sh
# Both entry points use the already prepared, offline Apple Silicon environment.
set -eu
cd "$(dirname "$0")"
exec sh ./run_offline.sh
