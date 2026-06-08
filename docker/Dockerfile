# syntax=docker/dockerfile:1
#
# Reproducible image for the KV cache benchmarks and prototype.
#
# The base image carries the pinned, tested vLLM + CUDA + PyTorch stack. Its
# CUDA build must match the host driver, so VLLM_TAG is overridable per hardware
# family. See docs/runbook.md for the driver -> CUDA -> tag matrix.
ARG VLLM_TAG=v0.22.0
FROM vllm/vllm-openai:${VLLM_TAG}

# Blackwell (sm_120): FlashInfer's JIT sampler aborts engine init. The native
# sampler is equivalent for greedy benchmarking and harmless on other GPUs, so
# we disable it by default everywhere.
ENV VLLM_USE_FLASHINFER_SAMPLER=0 \
    PYTHONUNBUFFERED=1 \
    PY=python3 \
    VLLM_BIN=vllm \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

# Harness and dev tooling, pinned. Installed before the source copy so layer
# caching survives code edits.
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

# Stage 3 only: Mooncake distributed KV cache. Off by default to keep the
# baseline image minimal and fast to build.
#   docker build --build-arg INSTALL_MOONCAKE=1 ...
ARG INSTALL_MOONCAKE=0
RUN if [ "$INSTALL_MOONCAKE" = "1" ]; then \
        pip install --no-cache-dir mooncake-transfer-engine; \
    fi

# The serve scripts default *_BIN to .venv paths for the host-virtualenv flow.
# In the container the tools are on PATH, so point the master binary there, the
# same way VLLM_BIN is set above. Placed after the installs so the pip layers
# stay cached on rebuild.
ENV MASTER_BIN=mooncake_master

COPY . .

# The vLLM base sets ENTRYPOINT to `vllm serve`; we drive it through our own
# scripts, so reset to a shell. Default command runs the baseline server.
ENTRYPOINT ["/bin/bash"]
CMD ["-lc", "bash scripts/serve_baseline.sh"]
