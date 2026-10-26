# Single image for all roles (orchestrator + 4 node services).
# The role is selected via the container command — see docker-compose.yml.
# Compatible with both `podman build` and `docker build`.
FROM python:3.11-slim

WORKDIR /app

# Copy metadata first for better layer caching, then the source.
COPY pyproject.toml README.md ./
COPY src ./src

# The project itself (runtime deps) plus pytest, which the tester container
# needs to execute the generated test suites.
RUN pip install --no-cache-dir . pytest

ENV PYTHONUNBUFFERED=1 \
    AIDEVTEAM_WORKSPACE_ROOT=/workspace

EXPOSE 8000

# Default: run the planner node on port 8000. Every node service listens on
# the same port inside its own container; docker-compose.yml overrides the
# command per service.
CMD ["python", "-m", "aidevteam.node_server", "planner", "8000"]
