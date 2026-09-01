FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# Sources are copied before the install because pyproject lists its packages
# explicitly, so they have to exist at build time. Installing from pyproject
# rather than a hand-written list keeps the image and the declared
# dependencies from drifting apart.
COPY pyproject.toml README.md ./
COPY agentserve/ ./agentserve/
COPY bench/ ./bench/
RUN pip install --no-cache-dir .

RUN adduser --disabled-password --gecos "" --uid 10001 agentserve
USER 10001

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8000/health').status_code==200 else 1)"

# One worker: the in-process replica cache state is per-process, so multiple
# uvicorn workers would each keep a divergent view. Scale with pods + Redis.
CMD ["uvicorn", "agentserve.gateway:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--timeout-graceful-shutdown", "30"]
