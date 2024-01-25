package main

import (
	"database/sql"
	"errors"
	"fmt"
	"github.com/golang-migrate/migrate/v4"
	"github.com/golang-migrate/migrate/v4/database/postgres"
	_ "github.com/golang-migrate/migrate/v4/source/file"
	_ "github.com/lib/pq"
	"log/slog"
	"os"
)

type Database struct {
	*sql.DB
}

type logger struct {
	*slog.Logger
}

func (l logger) Verbose() bool {
	return true
}
func (l logger) Printf(format string, v ...any) {
	l.Info(format, v)
}

type dbConfig struct {
	Host     string
	Port     string
	User     string
	Password string
	Name     string
}

func getDatabase(c dbConfig) (Database, error) {
	db, err := sql.Open(
		"postgres",
		fmt.Sprintf(
			"host=%s port=%s user=%s password=%s dbname=%s application_name=%s sslmode=disable",
			c.Host,
			c.Port,
			c.User,
			c.Password,
			c.Name,
			"velour",
		),
	)
	if err != nil {
		return Database{}, fmt.Errorf("error opening database: %w", err)
	}
	return Database{db}, nil
}

// TODO why is go-migrate throwing bad key errors? make a PR
var slogger = slog.New(slog.NewJSONHandler(os.Stdout, nil))

func (d *Database) MigrateUp(sqlPath string) error {
	driver, err := postgres.WithInstance(d.DB, &postgres.Config{MigrationsTable: "velour_schema_migrations"})
	if err != nil {
		return fmt.Errorf("error configuring driver for migration: %w", err)
	}
	migrateInstance, err := migrate.NewWithDatabaseInstance(sqlPath, "velour", driver)
	if err != nil {
		return fmt.Errorf("error creating new db migration instance: %w", err)
	}
	migrateInstance.Log = logger{slogger}
	// Close the migration connection and wrap existing error
	defer func() {
		sourceErr, dbServerErr := migrateInstance.Close()
		if sourceErr != nil {
			err = fmt.Errorf("%w err closing source %s", err, sourceErr.Error())
			return
		}
		if dbServerErr != nil {
			err = fmt.Errorf("%w err closing database  %s", err, dbServerErr.Error())
			return
		}
		slogger.Info("Successfully closed migration connections")
	}()
	err = migrateInstance.Up()
	if err != nil {
		if !errors.Is(err, migrate.ErrNoChange) {
			return fmt.Errorf("error applying migration: %w", err)
		}
		slogger.Info("No Change Detected. Skipping Migrations.")
	}

	slogger.Info("Migrations Complete.")
	return nil
}

func getConfigFromEnv() dbConfig {
	// TODO add logging for defaults
	host, found := os.LookupEnv("POSTGRES_HOST")
	if !found {
		panic("POSTGRES_HOST must be provided as an environment variable")
	}
	port, found := os.LookupEnv("POSTGRES_PORT")
	if !found {
		port = "5432"
	}
	user, found := os.LookupEnv("POSTGRES_USER")
	if !found {
		user = "postgres"
	}
	password, found := os.LookupEnv("POSTGRES_PASSWORD")
	if !found {
		panic("POSTGRES_PASSWORD must be provided as an environment variable")
	}
	name, found := os.LookupEnv("POSTGRES_DB")
	if !found {
		// TODO should be velour?
		name = "velour"
	}
	return dbConfig{
		Host:     host,
		Port:     port,
		User:     user,
		Password: password,
		Name:     name,
	}
}

func main() {
	db, err := getDatabase(getConfigFromEnv())
	if err != nil {
		panic(fmt.Errorf("error connecting to the database for migrations %w", err))
	}
	err = db.MigrateUp("file://sql")
	if err != nil {
		panic(fmt.Errorf("error running database migrations %w", err))
	}
}
