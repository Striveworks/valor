FROM migrate/migrate
# todo convert to wget?
COPY . /migrations
WORKDIR /migrations
CMD ["migrate", "up", "-source", "file://sql" ,"-driver", "postgres"]
#CMD ["go", "run", "main.go"]
#migrate -source file://migrations/sql -database postgres://$POSTGRES_USERNAME:$POSTGRES_PASSWORD@$POSTGRES_HOST:$POSTGRES_PORT/$POSTGRES_DB\?sslmode=disable up
