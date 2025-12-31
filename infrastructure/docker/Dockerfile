# Multi-stage Dockerfile for Trading Platform
# Stage 1: Build
FROM python:3.13-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY packages/ packages/

# Install dependencies
RUN pip install --no-cache-dir -e .

# Stage 2: Runtime
FROM python:3.13-slim as runtime

WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 trader

# Copy from builder
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /app /app

# Create data directories
RUN mkdir -p /app/data /app/logs && chown -R trader:trader /app

USER trader

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import packages.core" || exit 1

# Default command
CMD ["python", "-m", "packages.core.main", "--paper"]
