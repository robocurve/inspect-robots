# Build environment for Anthropic's OSS Scanner (https://red.anthropic.com/oss-scanner).
# The scanner builds this with the repository as the build context, then runs its
# agents in the image with no network access, so every dependency is installed here.
FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/src/.venv \
    PATH=/src/.venv/bin:$PATH \
    # hatch-vcs derives the version from git tags; pin one in case .git is absent.
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+oss-scanner

# ffmpeg backs `inspect-robots video` and report MP4s; libportaudio2 backs the
# voice plugin's sounddevice import.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git ffmpeg libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# scanner contract: the checkout lives inside the image, at /src
COPY . /src
WORKDIR /src

# Core plus every in-repo plugin, editable, with dev tools, exactly as CI installs them.
RUN uv sync --locked --all-packages --extra dev

# Smoke-test the image without failing the build on a test failure. Plugin suites run
# one at a time, as in CI, because their tests/ helper modules share names.
RUN (python -m pytest -q -p no:cacheprovider tests || echo "WARNING: core tests failed") \
    && for plugin in plugins/*/; do \
         (python -m pytest -q -p no:cacheprovider "${plugin}tests" \
           || echo "WARNING: ${plugin} tests failed"); \
       done
