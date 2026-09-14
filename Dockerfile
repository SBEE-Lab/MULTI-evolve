# syntax=docker/dockerfile:1@sha256:ecfaec9ed6d810b56388c508f4121597bfbba70d41a6dfeee4d8cad5f295fc32

FROM ghcr.io/astral-sh/uv:0.12.11@sha256:79c6f4776b851471cc73b7d21d0cc834bb94383c292e83640d27eff512864df7 AS uv

FROM registry.sjanglab.org/sjanglab/pytorch-runtime:2.6.0-cuda12.4-cudnn9-runtime@sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee AS builder

COPY --from=uv /uv /usr/local/bin/uv

WORKDIR /opt/multievolve

ENV UV_COMPILE_BYTECODE=1 \
    UV_CONCURRENT_DOWNLOADS=1 \
    UV_HTTP_TIMEOUT=600 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --python /opt/conda/bin/python --system-site-packages .venv \
    && uv sync --locked --no-dev --no-install-project \
        --no-install-package nvidia-cublas-cu12 \
        --no-install-package nvidia-cuda-cupti-cu12 \
        --no-install-package nvidia-cuda-nvrtc-cu12 \
        --no-install-package nvidia-cuda-runtime-cu12 \
        --no-install-package nvidia-cudnn-cu12 \
        --no-install-package nvidia-cufft-cu12 \
        --no-install-package nvidia-curand-cu12 \
        --no-install-package nvidia-cusolver-cu12 \
        --no-install-package nvidia-cusparse-cu12 \
        --no-install-package nvidia-cusparselt-cu12 \
        --no-install-package nvidia-nccl-cu12 \
        --no-install-package nvidia-nvjitlink-cu12 \
        --no-install-package nvidia-nvtx-cu12 \
        --no-install-package torch \
        --no-install-package triton

COPY LICENSE MANIFEST.in README.md setup.py ./
COPY multievolve/ multievolve/
COPY scripts/ scripts/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python --no-deps .

FROM registry.sjanglab.org/sjanglab/pytorch-runtime:2.6.0-cuda12.4-cudnn9-runtime@sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee AS runtime

ARG VCS_REF=unknown

LABEL org.opencontainers.image.base.name="registry.sjanglab.org/sjanglab/pytorch-runtime:2.6.0-cuda12.4-cudnn9-runtime" \
      org.opencontainers.image.base.digest="sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee" \
      org.opencontainers.image.description="Locked runtime for MULTI-evolve" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.source="https://github.com/SBEE-Lab/MULTI-evolve" \
      org.opencontainers.image.title="MULTI-evolve" \
      org.opencontainers.image.version="0.1.0"

RUN mkdir --mode=1777 /work

COPY --from=builder /opt/multievolve/.venv /opt/multievolve/.venv
COPY app.py /opt/multievolve/app.py
COPY .streamlit/config.toml /opt/multievolve/.streamlit/config.toml

ENV PATH="/opt/multievolve/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /work

CMD ["python"]
