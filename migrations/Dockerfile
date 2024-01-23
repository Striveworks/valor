FROM migrate/migrate
# todo convert to wget?
COPY . /migrations
WORKDIR /migrations
CMD ["migrate", "up", "-source", "file://sql" ,"-driver", "postgres"]
#CMD ["go", "run", "main.go"]
#migrate -source file://migrations/sql -database postgres://postgres:password@localhost:5432/velour\?sslmode=disable up
