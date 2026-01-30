# Multi-stage build for the FraudShield FastAPI inference service.
#
# Stage 1 (builder)  installs build deps and the package wheel.
# Stage 2 (runtime)  copies just the venv + source over a slim Python base.
#
# The runtime image lands around 600 MB — most of it is LightGBM + SHAP +
# scikit-learn binaries. To trim further, swap to python:3.12-alpine and
# rebuild LightGBM from source. We don't bother here because the savings
# don't justify the wheel-rebuild time.

# ------------------------------ Stage 1 ------------------------------
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# System deps needed for compiling the small handful of source-only wheels
RUN apt-get update \
 && apt-get install --no-install-recommends -y \
        build-essential libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Install runtime deps first (cached layer until requirements change)
COPY requirements.txt ./
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install -r requirements.txt

# Then install our package from the workdir
COPY pyproject.toml README.md ./
COPY src ./src
RUN /opt/venv/bin/pip install --no-deps .

# ------------------------------ Stage 2 ------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    FRAUDSHIELD_MODELS_DIR=/app/models

# libgomp is needed at runtime by LightGBM
RUN apt-get update \
 && apt-get install --no-install-recommends -y libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --shell /usr/sbin/nologin --uid 1000 fraudshield

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv

# Trained artifacts get mounted at /app/models via volume, but ship the
# directory so the lifespan can find it without erroring on missing path.
RUN mkdir -p /app/models && chown -R fraudshield:fraudshield /app

USER fraudshield

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status==200 else 1)"

CMD ["uvicorn", "fraud_shield.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
