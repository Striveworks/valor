package main

import (
	"database/sql"
	"errors"
	"flag"
	"fmt"
	"github.com/golang-migrate/migrate/v4"
	"github.com/golang-migrate/migrate/v4/database/postgres"
	_ "github.com/golang-migrate/migrate/v4/source/file"
	_ "github.com/lib/pq"
	"log/slog"
)

type Database struct {
	*sql.DB
}

type logger struct {
	slog.Logger
}

func (l logger) Verbose() bool {
	return true
}
func (l logger) Printf(format string, v ...interface{}) {
	l.Info(format, v)
}

type dbConfig struct {
	Host     string
	Port     string
	User     string
	Password string
	Name     string
}

func getDatabase(c dbConfig) (*sql.DB, error) {
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
		return nil, fmt.Errorf("error opening database: %w", err)
	}
	return db, nil
}

var slogger = slog.Logger{}

func (d *Database) MigrateUp(sqlPath string) error {
	driver, err := postgres.WithInstance(d.DB, &postgres.Config{MigrationsTable: "models_catalog_schema_migrations"})
	if err != nil {
		return fmt.Errorf("error configuring driver for migration: %w", err)
	}
	migrateInstance, err := migrate.NewWithDatabaseInstance(sqlPath, "velour", driver)
	if err != nil {
		return fmt.Errorf("error creating new db migration instance: %w", err)
	}
	// TODO validate logging
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

//func migrate() {
//	migrationsDB, err := dao.GetMigrationDatabase(config.Config.Database)
//	if err != nil {
//		log.FatalErr(fmt.Errorf("error getting database: %w", err))
//	}
//	// Run migrations and close DB connection.
//	err = migrationsDB.MigrateUp("file://migrations")
//	if err != nil {
//		log.FatalErr(fmt.Errorf("error migrating database: %w", err))
//	}
//}

func main() {
	host := flag.String("host", "", "db host address")
	port := flag.String("port", "", "db port")
	user := flag.String("user", "", "postgres user")
	password := flag.String("password", "", "postgres password")
	name := flag.String("name", "", "database name")
	flag.Parse()

	//if host == nil {
	//
	//}
	//if port == nil {
	//
	//}
	//if user == nil {
	//
	//}
	//if password == nil {
	//
	//}
	//if name == nil {
	//
	//}

	dbConfig{
		Host:     host,
		Port:     port,
		User:     user,
		Password: password,
		Name:     name,
	}
	db, _ := getDatabase()
}
