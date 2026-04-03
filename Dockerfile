# Use a lightweight Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements first to speed up builds
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files into the container
COPY . .

# Expose port 7860 (Hugging Face standard)
EXPOSE 7860

# Command to run the server
# Note: We use 0.0.0.0 so it is reachable from outside the container
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]