#!/bin/bash

# Define your Docker image name
IMAGE_NAME="assignment4:a4"

# Build the Docker image
docker build -t $IMAGE_NAME .

# Check if the image was built successfully
if [ $? -eq 0 ]; then
  echo "Docker image $IMAGE_NAME built successfully."
else
  echo "Error: Failed to build Docker image $IMAGE_NAME."
  exit 1
fi

# Define the host directory where you want to save the trained models
HOST_MODEL_DIR="/MLOPS_23"

# Create the host directory if it doesn't exist
mkdir -p $HOST_MODEL_DIR

# Run the Docker container with the model volume
docker run -v $HOST_MODEL_DIR:/models $IMAGE_NAME

# Check if the container executed successfully
if [ $? -eq 0 ]; then
  echo "Docker container $IMAGE_NAME ran successfully."
else
  echo "Error: Docker container $IMAGE_NAME encountered an issue."
  exit 1
fi

