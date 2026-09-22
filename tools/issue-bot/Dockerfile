FROM docker.io/cloudflare/sandbox:0.12.9
RUN npm install -g @openai/codex@0.155.1 typescript@7.0.2
RUN apt-get update && apt-get install -y --no-install-recommends ripgrep procps && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python
RUN uv python install 3.11 && uv venv --python 3.11 /opt/issue-env && uv pip install --python /opt/issue-env/bin/python numpy==2.2.6 pytest==8.4.2 pytest-cov==6.3.0 hypothesis==6.140.0 pip==25.2 hatchling==1.27.0 hatch-vcs==0.5.0 hatch-fancy-pypi-readme==25.1.0 ruff==0.13.0 mypy==1.18.2
# Cached generic JS tools are available offline; project-specific dependencies may
# still be absent and must be reported as unverified, never claimed to pass.
RUN mkdir -p /opt/issue-js && cd /opt/issue-js && npm init -y && npm install --legacy-peer-deps --ignore-scripts --save-exact typescript@7.0.2 vitest@4.1.11 @types/node@26.6.2 && chmod -R a+rX /opt/issue-js
ENV PATH="/opt/issue-env/bin:/opt/issue-js/node_modules/.bin:$PATH"
COPY sandbox/issue_runner.py /opt/issue-runner.py
COPY sandbox/supervisor.py /opt/issue-supervisor.py
EXPOSE 8080

# Override the Sandbox image management server: only the fixed authenticated
# supervisor may run as root. There is no generic exec/file/terminal endpoint.
ENTRYPOINT ["/opt/issue-env/bin/python", "-I", "/opt/issue-supervisor.py"]
CMD []
