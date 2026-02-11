FROM python:3.12-slim AS base

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY *.py ./

# Create non-root user
RUN useradd --create-home --shell /bin/bash cliai
USER cliai

# Config and data directories (mount as volumes for persistence)
ENV XDG_CONFIG_HOME=/home/cliai/.config
ENV XDG_DATA_HOME=/home/cliai/.local/share

ENTRYPOINT ["python", "chat_cli.py"]
CMD ["chat"]
