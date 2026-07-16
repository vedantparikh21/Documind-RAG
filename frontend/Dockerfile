# Same base image as backend — consistency makes debugging easier
FROM python:3.11-slim

WORKDIR /app

# Same layer-caching pattern: requirements before source code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# Streamlit in Docker needs three extra flags that aren't needed locally:
#
# --server.address=0.0.0.0
#   Same reason as backend: accept connections from outside the container.
#
# --server.headless=true
#   Locally, Streamlit tries to open your browser automatically.
#   Inside a container there is no browser — this flag disables that attempt.
#   Without it, Streamlit might throw errors or hang on startup.
#
# --server.fileWatcherType=none
#   Locally, Streamlit watches your files for changes and auto-reruns.
#   In a container the files never change at runtime, so this watcher
#   just wastes CPU. Disable it.
CMD ["streamlit", "run", "app.py","--server.address=0.0.0.0","--server.port=8501","--server.headless=true","--server.fileWatcherType=none"]