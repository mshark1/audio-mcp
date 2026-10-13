# Runs the audio-mcp server over stdio, e.g. for directories (Glama) that introspect MCP servers in a container.
#   docker build -t audio-mcp .
#   docker run -i --rm -v "$PWD/audio:/audio" -e AUDIO_MCP_ALLOWED_DIRS=/audio audio-mcp
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install . && useradd --create-home --uid 1000 mcp
USER mcp

ENTRYPOINT ["audio-mcp"]
