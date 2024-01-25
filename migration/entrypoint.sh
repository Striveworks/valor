#!/bin/sh

export POSTGRES_USER="$POSTGRES_USER"
export POSTGRES_PASSWORD="$POSTGRES_PASSWORD"
export POSTGRES_HOST="$POSTGRES_HOST"
export POSTGRES_PORT="$POSTGRES_PORT"
export POSTGRES_DB="$POSTGRES_DB"

MAX_RETRIES=10
wait_for_postgres() {
  retries=0
  until nc -z -w 1 $POSTGRES_HOST $POSTGRES_PORT || [ $retries -eq $MAX_RETRIES ]; do
    echo "Waiting for PostgreSQL to be ready... (Retry $((retries+1)) of $MAX_RETRIES)"
    sleep 1
    retries=$((retries+1))
  done

  if [ $retries -eq $MAX_RETRIES ]; then
    echo "Max retries reached. PostgreSQL might not be ready. Exiting..."
    exit 1
  fi

  echo "PostgreSQL is ready!"
}
migrate -path /migrations/sql -database postgres://$POSTGRES_USER:$POSTGRES_PASSWORD@$POSTGRES_HOST:$POSTGRES_PORT/$POSTGRES_DB?sslmode=disable "$@"
