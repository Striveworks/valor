ARG BASE_IMAGE_TAG=latest

FROM postgis/postgis:$BASE_IMAGE_TAG as base-image

FROM base-image as basic-deps

RUN apt-get update && \
	apt-get install -y --no-install-recommends \
		ca-certificates \
		curl

RUN apt-get update && \
	apt-get install -y --no-install-recommends \
		postgresql-$PG_MAJOR-pgvector && \
	apt-get purge -y --auto-remove && \
	rm -rf /var/lib/apt/lists/*