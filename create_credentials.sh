#!/bin/bash

# Prompt for GitHub username
read -p "Enter your GitHub username: " github_username

# Prompt for GitHub token
read -sp "Enter your GitHub token: " github_token
echo

# Create Kubernetes secret for GitHub credentials
kubectl create secret docker-registry ghcr-secret \
    --docker-server=ghcr.io \
    --docker-username="$github_username" \
    --docker-password="$github_token" \
    --docker-email=""

# Prompt for Postgres Credentials

# Default values
DEFAULT_PG_USER="user"
DEFAULT_PG_PASSWORD="password"

# Prompt user for input
read -p "Enter PostgresDB username [${DEFAULT_PG_USER}]: " PG_USER
PG_USER="${PG_USER:-${DEFAULT_PG_USER}}"

read -p "Enter PostgresDB password [${DEFAULT_PG_PASSWORD}]: " PG_PASSWORD
PG_PASSWORD="${PG_PASSWORD:-${DEFAULT_PG_PASSWORD}}"

# Create credentials for PostgresDB
kubectl create secret generic postgres-credentials \
  --from-literal=POSTGRES_PASSWORD=${PG_PASSWORD} \
  --from-literal=POSTGRES_USER=${PG_USER}
