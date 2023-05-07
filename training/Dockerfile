FROM prefecthq/prefect:2-python3.9

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
WORKDIR /app

COPY requirements.txt .

# Install production dependencies.
RUN pip install -r requirements.txt


COPY . .

# Run the web service on container startup using gunicorn

# CMD ["python", "training/docker_deploy.py"]
# CMD ["python", "training/training.py"]
