-- this migration goes  through dataset names and model names and replaces commas with underscores
-- if the resulting name happens to exist, it adds underscores until it gets a name that doesn't
-- this will update the following tables: Model, Dataset, Evaluation

-- Function to get a unique name by adding underscores
CREATE OR REPLACE FUNCTION get_unique_name(base_name TEXT, table_name TEXT)
RETURNS TEXT AS $$
DECLARE
    unique_name TEXT := base_name;
    name_exists INT;
BEGIN
    EXECUTE format('SELECT COUNT(*) FROM %I WHERE name = $1', table_name) INTO name_exists USING unique_name;

    WHILE name_exists > 0 LOOP
        unique_name := unique_name || '_';
        EXECUTE format('SELECT COUNT(*) FROM %I WHERE name = $1', table_name) INTO name_exists USING unique_name;
    END LOOP;

    RETURN unique_name;
END;
$$ LANGUAGE plpgsql;

DO $$
DECLARE
    old_name TEXT;
    new_name TEXT;
BEGIN
    FOR old_name IN SELECT name FROM model WHERE POSITION(',' IN name) > 0 LOOP
        new_name := get_unique_name(REPLACE(old_name, ',', '_'), 'model');

        UPDATE Model SET name = new_name WHERE name = old_name;

        UPDATE Evaluation SET model_name = new_name WHERE model_name = old_name;
    END LOOP;
END;
$$;

DO $$
DECLARE
    old_name TEXT;
    new_name TEXT;
BEGIN
    FOR old_name IN SELECT name FROM Dataset WHERE POSITION(',' IN name) > 0 LOOP
        new_name := get_unique_name(REPLACE(old_name, ',', '_'), 'dataset');

        UPDATE Dataset SET name = new_name WHERE name = old_name;

        UPDATE Evaluation
        SET dataset_names = (
            SELECT jsonb_agg(
                CASE
                    WHEN elem = old_name THEN new_name
                    ELSE elem
                END
            )
            FROM jsonb_array_elements_text(dataset_names) AS elem
        )
        WHERE dataset_names @> jsonb_build_array(old_name);
    END LOOP;
END;
$$;

DROP FUNCTION get_unique_name;