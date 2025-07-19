#!/usr/bin/env bash
set -e -x

exec moshi-server worker --config /docker.toml -p ${PORT:-8080}