ARG BASE_IMAGE_TAG=latest
FROM bitnami/postgresql:$BASE_IMAGE_TAG as base-image
ARG PG_MAJOR=16
ARG PG_VECTOR=0.6.2
USER 0
RUN apt-get update && \
		apt-mark hold locales && \
		apt-get install -y --no-install-recommends wget build-essential postgresql-server-dev-${PG_MAJOR} && \
		wget https://github.com/pgvector/pgvector/archive/refs/tags/v${PG_VECTOR}.tar.gz -P /tmp && \
		mkdir -p /tmp/pgvector && \
        tar -xvzf /tmp/v${PG_VECTOR}.tar.gz --strip-components=1 -C /tmp/pgvector && \
		cd /tmp/pgvector && \
		make clean && \
		make OPTFLAGS="" && \
		make install && \
		mkdir /usr/share/doc/pgvector && \
		cp LICENSE README.md /usr/share/doc/pgvector && \
		rm -r /tmp/pgvector && \
		apt-get remove -y build-essential postgresql-server-dev-$PG_MAJOR && \
		apt-get autoremove -y && \
		apt-mark unhold locales && \
		rm -rf /var/lib/apt/lists/*
