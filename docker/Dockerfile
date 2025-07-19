FROM rust:1.88-slim as builder
ARG COMMIT="aa4fb9a7e328beddaa607f84ad1f0a86183aa5cc"

RUN export DEBIAN_FRONTEND=noninteractive=1 \
    && apt-get update \
    && apt-get install -y \
    curl \
    build-essential \
    ca-certificates \
    libssl-dev \
    git \
    pkg-config \
    cmake \
    wget \
    openssh-client \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

ENV ARTIFACT /usr/local
RUN mkdir -p "${ARTIFACT}"
WORKDIR "${ARTIFACT}"
## python
# Python image is configured for `/usr/local`
# XXX: moshi-server's pyproject.toml specifies this version
COPY --from=python:3.12.8-slim /usr/local/ /usr/local

## git checkout
RUN cd "${TMPDIR:-/tmp}" \
    && git clone --sparse https://github.com/kyutai-labs/moshi.git \
    && cd moshi \
    && git sparse-checkout set rust \
    && git reset --hard "${COMMIT}" \
    && git checkout . \
    && cd -

## uv python depends
RUN cp "${TMPDIR:-/tmp}/moshi/rust/moshi-server/uv.lock" "${TMPDIR:-/tmp}/moshi/rust/moshi-server/pyproject.toml" .
COPY --from=ghcr.io/astral-sh/uv:0.7.2 /uv /uvx ${ARTIFACT}/bin/
ENV UV_INSTALL_DIR "${ARTIFACT}"
ENV UV_PROJECT_ENVIRONMENT "${ARTIFACT}"
ENV UV_BREAK_SYSTEM_PACKAGES true
ENV UV_NO_MANAGED_PYTHON true
RUN uv sync

## cargo build
RUN cargo install \
    --path "${TMPDIR:-/tmp}/moshi/rust/moshi-server" \
    --root "${ARTIFACT}/"

FROM nvidia/cuda:12.5.1-runtime-ubuntu22.04 AS base
COPY --from=builder /usr/local /usr/local
RUN ldconfig

COPY configs/docker.toml /docker.toml
COPY entrypoint.sh /entrypoint.sh

# FIXME: Huggingface tools are not respecting environment variables,
#   cheat a little by making a anonymous volume the home directory
VOLUME [ "/data" ]
ENV HF_HOME /data
RUN useradd user -d /data \
    && chown user:user /data
USER user

# When starting the container for the first time, we need to compile and download
# everything, so disregarding healthcheck failure for 10 minutes is fine.
# We have a volume storing the build cache, so subsequent starts will be faster.
ENV PORT 8080
EXPOSE $PORT
HEALTHCHECK --start-period=10m \
    CMD curl --fail http://localhost:$PORT/api/build_info || exit 1

ENTRYPOINT "/entrypoint.sh"