# CPU-only evaluator image.  It contains no inference server or model runtime.
FROM python:3.11.11-slim-bookworm@sha256:081075da77b2b55c23c088251026fb69a7b2bf92471e491ff5fd75c192fd38e5

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    QUALITY_EVAL_SANDBOX=1 \
    QUALITY_INPUT_ROOT=/input \
    HOME=/tmp \
    HF_DATASETS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# These are the only evaluator distributions intentionally installed.  Their
# transitive dependencies are resolved by pip for the pinned Python image.
RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.6.0+cpu" \
    && python -m pip install --no-cache-dir --disable-pip-version-check \
      "lm-eval==0.4.13" \
      "evalplus==0.3.1" \
    && python - <<'PY'
from importlib.metadata import version
assert version("lm-eval") == "0.4.13"
assert version("evalplus") == "0.3.1"
PY

ENV NLTK_DATA=/opt/nltk_data
RUN python -m pip install "lm-eval[ifeval]==0.4.13" \
    && python - <<'PY'
import nltk
import torch
assert torch.__version__ == "2.6.0+cpu" and torch.version.cuda is None
nltk.download("punkt_tab", download_dir="/opt/nltk_data", raise_on_error=True)
from lm_eval.tasks.ifeval import utils
PY

RUN groupadd --system --gid 65532 quality \
    && useradd --system --uid 65532 --gid 65532 --home-dir /tmp --no-create-home quality

WORKDIR /opt/quality
COPY --chown=65532:65532 quality.py /opt/quality/quality.py
USER 65532:65532

ENTRYPOINT ["python", "/opt/quality/quality.py"]
