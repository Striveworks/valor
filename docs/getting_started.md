# Getting Started

Velour is a centralized evaluation store which makes it easy to measure, explore, and rank model performance. For an overview of what Velour is and why it's important, please refer to our [high-level overview](index.md).

On this page, we'll describe how to get up and running with Velour.

## 1. Install Docker

As a first step, be sure your machine has Docker installed. [Click here](https://docs.docker.com/engine/install/) for basic installation instructions.


## 2. Clone the repo and open the directory

Choose a file in which to store Velour, then run:

```shell
git clone https://github.com/striveworks/velour
cd velour
```


## 3. Start services

There are four ways to start the Velour API service.

### a. Docker Compose

The fastest way to test the API and Python client is via Docker Compose. Start by setting the environment variable `POSTGRES_PASSWORD` to your liking, then start Docker and build the container:

```shell
export POSTGRES_PASSWORD="my_password"
docker compose up
```

### b. Running the Server Locally

Alternatively, you may want to run the API service locally when debugging or developing Velour. To start the service in your Terminal, you can run:

```shell
pip install api # Install the API in your python environment

export POSTGRES_PASSWORD=password
export POSTGRES_HOST=localhost
make start-postgis # Start the postgis service in Docker
make start-redis # Start the redis service in Docker
make start-server # Start the API service locally
```

### c. Helm Chart

When deploying Velour on k8s via Helm, you can use our pre-built chart using the following commands:

```shell
helm repo add velour https://striveworks.github.io/velour-charts/
helm install velour velour/velour
# Velour should now be avaiable at velour.namespace.svc.local
```

### d. Pre-Built Docker Container

Finally, you can download the Velour image from GitHub's Container registry at `ghcr.io/striveworks/velour/velour-service`.


## 4. Use Velour

There's two ways to access Velour: by leveraging our Python client, or by calling our REST endpoints directly.

### 4a. Using the Python client

Please see our ["Getting Started"](#TODO) notebook for a working example of our Python client.


### 4b. Using API endpoints
You can also leverage Velour's API without using the Python client. [Click here](endpoints.md) to read up on all of our API endpoints.


# Next Steps

For more examples, we'd recommend reviewing our [sample notebooks on GitHub](#TODO). For more detailed explainations of Velour's technical underpinnings, see our [technical concepts guide](technical_concepts.md).