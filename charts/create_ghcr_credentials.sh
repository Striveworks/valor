#!/bin/bash

# Prompt for GitHub username
read -p "Enter your GitHub username: " github_username

# Prompt for GitHub token
read -sp "Enter your GitHub token: " github_token
echo

# Create Kubernetes secret for GitHub credentials
kubectl create secret docker-registry velour-ghcr-secret \
    --docker-server=ghcr.io \
    --docker-username="$github_username" \
    --docker-password="$github_token" \
    --docker-email=""
