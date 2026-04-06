# Use a lightweight Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system-level curl (required for the Healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements first to speed up builds
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Set Python path so 'server.app' can find 'models' and 'scenarios'
ENV PYTHONPATH="/app"

# Expose port 7860 (Hugging Face standard)
EXPOSE 7860

# Mandatory Healthcheck for the OpenEnv spec
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1

# Launch the server using the module approach
CMD ["python", "-m", "server.app"]