#!/bin/sh

export POSTGRES_USERNAME="$POSTGRES_USERNAME"
export POSTGRES_PASSWORD="$POSTGRES_PASSWORD"
export POSTGRES_HOST="$POSTGRES_HOST"
export POSTGRES_PORT="$POSTGRES_PORT"
export POSTGRES_DB="$POSTGRES_DB"
export POSTGRES_SSLMODE="${POSTGRES_SSLMODE:-disable}"

MAX_RETRIES=10
WAIT_SECONDS=3


wait_for_postgres() {
  retries=0
  until PGPASSWORD=$POSTGRES_PASSWORD psql -c "select 1" "sslmode=$POSTGRES_SSLMODE dbname=$POSTGRES_DB host=$POSTGRES_HOST user=$POSTGRES_USERNAME" >& /dev/null || [ $retries -eq $MAX_RETRIES ]; do
    echo "Waiting for PostgreSQL to be ready... (Retry $((retries+1)) of $MAX_RETRIES)"
    sleep $WAIT_SECONDS
    retries=$((retries+1))
  done

  if [ $retries -eq $MAX_RETRIES ]; then
    echo "Max retries reached. PostgreSQL might not be ready. Exiting..."
    exit 1
  fi

  echo "PostgreSQL is ready."
}

wait_for_postgres

migrate -path /migrations/sql -database "postgres://${POSTGRES_USERNAME}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}?sslmode=${POSTGRES_SSLMODE}&application_name=valor_migrations" "$@"

echo "Migration complete."
