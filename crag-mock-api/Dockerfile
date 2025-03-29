# Use an official lightweight Python image.
# 3.10-slim is a good balance between size and compatibility
FROM python:3.10-slim

# Set environment variables
# Python won't try to write .pyc files on the import of source modules
ENV PYTHONDONTWRITEBYTECODE 1
# Python buffers stdout and stderr by default, setting this to false to avoid that
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Command to run the uvicorn server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
