FROM migrate/migrate
COPY . /migrations
WORKDIR /migrations
RUN which migrate
ARG POSTGRES_USER
ARG POSTGRES_PASSWORD
ARG POSTGRES_HOST
ARG POSTGRES_PORT
ARG POSTGRES_DB
#CMD ["migrate", "-path","/migrations/sql", "-database", "postgres://$POSTGRES_USER:$POSTGRES_PASSWORD@$POSTGRES_HOST:$POSTGRES_PORT/$POSTGRES_DB?sslmode=disable", "up"]
#CMD ["migrate", "-path", "/migrations/sql", "-database", "postgres://postgres:password@db:5432/velour?sslmode=disable", "up"]
#CMD ["sleep", "100000000"]
#ENTRYPOINT "up"

#migrate -source file://migrations/sql -database postgres://postgres:password@db:5432/velour?sslmode=disable up
#zsh: no matches found: postgres://:@:/?sslmode=disable
#ENTRYPOINT ["migrate","-path","/migrations/sql", "-database", "postgres://$POSTGRES_USER:$POSTGRES_PASSWORD@$POSTGRES_HOST:$POSTGRES_PORT/$POSTGRES_DB?sslmode=disable"]
ENTRYPOINT ["migrate","-path","/migrations/sql", "-database", "postgres://postgres:password@db:5432/velour?sslmode=disable"]
CMD ["up"]