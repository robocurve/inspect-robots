FROM docker.io/cloudflare/sandbox:0.12.9
RUN npm install -g @openai/codex@0.155.1
RUN apt-get update && apt-get install -y --no-install-recommends ripgrep && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python
RUN uv python install 3.11 && uv venv --python 3.11 /opt/review-env && uv pip install --python /opt/review-env/bin/python numpy==2.2.6 pytest==8.4.2 pytest-cov==6.3.0 hypothesis==6.140.0 pip==25.2 hatchling==1.27.0 hatch-vcs==0.5.0 hatch-fancy-pypi-readme==25.1.0 httpx==0.28.1 websockets==16.0 mypy==2.1.0 ruff==0.15.20
ENV PATH="/opt/review-env/bin:$PATH"
COPY sandbox/codex_runner.py /opt/codex-review.py
COPY sandbox/check_isolation.py /opt/check-isolation.py
EXPOSE 8080
