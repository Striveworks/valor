FROM docker.io/migrate/migrate
COPY . /migrations
WORKDIR /migrations

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

RUN apk --update add postgresql-client

ENTRYPOINT ["/entrypoint.sh"]
CMD ["up", "1"]
USER 65532:65532
