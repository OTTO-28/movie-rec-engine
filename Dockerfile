# --- STAGE 1: BUILDER ---
# CHANGED: Updated to python:3.11-slim
FROM python:3.11-slim AS builder

# Install uv binary from Astral's image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Enable bytecode compilation & copy mode
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# 1. Install dependencies (Cached Layer)
COPY pyproject.toml uv.lock ./

# Sync dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# 2. Install the project
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


# --- STAGE 2: RUNTIME ---
# CHANGED: Updated to python:3.11-slim to match builder
FROM python:3.11-slim

WORKDIR /app

# Copy the virtual environment from the builder
COPY --from=builder /app/.venv /app/.venv

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code
COPY . .

# Run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]