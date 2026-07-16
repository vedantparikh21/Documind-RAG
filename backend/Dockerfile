# ── Base image ────────────────────────────────────────────────────────────────
# python:3.11-slim is a minimal Debian Linux with Python 3.11 pre-installed.
# "slim" removes documentation, test files, and unused OS packages.
# Result: ~130MB vs ~900MB for the full python image. Always use slim.
FROM python:3.11-slim

# ── Working directory ──────────────────────────────────────────────────────────
# All subsequent commands run relative to /app inside the container.
# This is where your code will live.
WORKDIR /app

# ── Install dependencies BEFORE copying source code ───────────────────────────
# This is the most important Docker optimization: layer caching.
#
# Docker builds images in layers — one layer per instruction.
# If a layer's input hasn't changed, Docker reuses the cached layer (instant).
# If it has changed, Docker rebuilds that layer AND every layer after it.
#
# By copying requirements.txt first and running pip install before copying
# your source code, we ensure:
#   - Code changes (app.py edits) → only the COPY . . layer rebuilds (fast)
#   - requirements.txt changes → pip install rebuilds (slow, but necessary)
#
# Wrong order (COPY . . first) means every tiny code change triggers
# a full pip install — could waste 5 minutes per rebuild.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# --no-cache-dir: don't store pip's download cache inside the image.
# Saves ~50MB. We won't be running pip again inside this container.

# ── Copy source code ───────────────────────────────────────────────────────────
# The dot means "copy everything from backend/ into /app inside the container"
# .dockerignore controls what gets excluded (see backend/.dockerignore)
COPY . .

# ── Create runtime directories ─────────────────────────────────────────────────
# faiss_index/ and uploads/ are written to at runtime by the app.
# We create them here so they exist even before any volumes are mounted.
RUN mkdir -p faiss_index uploads

# ── Expose port ────────────────────────────────────────────────────────────────
# EXPOSE is documentation — it tells Docker (and humans) which port this
# container listens on. It doesn't actually open the port; that's done
# in docker-compose.yml with the "ports" mapping.
EXPOSE 8000

# ── Startup command ────────────────────────────────────────────────────────────
# CMD is the command that runs when the container starts.
# Key difference from local dev: --host 0.0.0.0 (not 127.0.0.1)
# 
# 127.0.0.1 = "only accept connections from this machine itself"
# 0.0.0.0   = "accept connections from anywhere" (other containers, host, internet)
#
# Without 0.0.0.0, the frontend container cannot reach the backend container
# even though they're on the same Docker network.
#
# No --reload flag — that's for development only. In a container, the code
# doesn't change at runtime, so hot-reload is unnecessary overhead.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]