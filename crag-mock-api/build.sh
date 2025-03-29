#!/bin/bash

# This script is designed to automate the process of building and optionally pushing
# Docker images for a specific project. It sets up the necessary environment variables,
# computes the Docker tags based on the git commit hash, builds the Docker images, and
# optionally pushes them to a Docker registry.

# Define environment variables for Docker configuration
export DOCKER_USERNAME="aicrowd"                   # Docker Hub username or organization name
export DOCKER_REPO="kdd-cup-24-crag-mock-api"              # Repository name on Docker Hub
export BASE_VERSION="v1"                           # Base version of the Docker image

# Retrieve the latest commit hash of the git repository and shorten it
# This hash is used to create a unique Docker tag for the build
export LATEST_COMMIT_HASH="$(git rev-parse --short HEAD)"

# Combine the base version with the latest commit hash to form a granular Docker tag
export DOCKER_TAG_GRANULAR="${BASE_VERSION}-${LATEST_COMMIT_HASH}"

# Construct the base and granular Docker image names using the defined variables
# The base image name uses only the base version as its tag
# The granular image name includes the detailed version tag with the commit hash
export DOCKER_IMAGE_NAME_BASE="${DOCKER_USERNAME}/${DOCKER_REPO}:${BASE_VERSION}"
export DOCKER_IMAGE_NAME_GRANULAR="${DOCKER_USERNAME}/${DOCKER_REPO}:${DOCKER_TAG_GRANULAR}"

# Echo the constructed Docker image names for logging and verification purposes
echo "Docker Image Name (Base): ${DOCKER_IMAGE_NAME_BASE}"
echo "Docker Image Name (Granular): ${DOCKER_IMAGE_NAME_GRANULAR}"

# Build the Docker images using the tags defined above
# The same Docker context is used to build both images, ensuring they are identical
# but tagged differently for different use cases (e.g., stable vs. latest development)
DOCKER_BUILDKIT=1 docker build -t "${DOCKER_IMAGE_NAME_BASE}" -t "${DOCKER_IMAGE_NAME_GRANULAR}" .

# Check if the first script argument is "push"
# If so, both the base and granular images are pushed to the Docker registry
# This step requires that Docker authentication has been set up beforehand
if [ "$1" == "push" ]; then
  docker push "${DOCKER_IMAGE_NAME_BASE}"
  docker push "${DOCKER_IMAGE_NAME_GRANULAR}"
fi
