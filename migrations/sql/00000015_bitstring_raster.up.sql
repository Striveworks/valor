ALTER TABLE annotation ADD COLUMN bitmask bit varying;

CREATE OR REPLACE FUNCTION raster_to_bitstring(input_raster raster)
RETURNS text AS
$$
DECLARE
    width INTEGER;
    height INTEGER;
    bitstring TEXT := '';
    val INTEGER;
    pixelval INTEGER;
BEGIN
    -- Get the width and height of the raster
    SELECT ST_Width(input_raster), ST_Height(input_raster) INTO width, height;

    -- Iterate through each row and column of the raster
    FOR i IN 1..height LOOP
        FOR j IN 1..width LOOP
            -- Get the pixel value at position (i, j)
            pixelval := ST_Value(input_raster, i, j);

            -- 1BB raster has values of 0 or 1, so just append it to the bitstring
            IF pixelval IS NOT NULL THEN
                val := pixelval::integer;
                bitstring := bitstring || val::text;
            ELSE
                bitstring := bitstring || '0'; -- or handle NULL as needed
            END IF;
        END LOOP;
    END LOOP;

    RETURN bitstring;
END;
$$
LANGUAGE plpgsql;
