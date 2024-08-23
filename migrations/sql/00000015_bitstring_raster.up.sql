CREATE OR REPLACE FUNCTION raster_to_bitstring(input_raster raster)
RETURNS BIT VARYING AS
$$
DECLARE
    width INTEGER;
    height INTEGER;
    bitstring BIT VARYING := B'';
    val INTEGER;
    pixelval INTEGER;
BEGIN
    -- Get the width and height of the raster
    SELECT ST_Width(input_raster), ST_Height(input_raster) INTO width, height;

    -- Iterate through each row and column of the raster
    FOR i IN 1..width LOOP
        FOR j IN 1..height LOOP
            -- Get the pixel value at position (j, i)
            pixelval := ST_Value(input_raster, j, i);

            -- 1BB raster has values of 0 or 1, so just append it to the bitstring
            IF pixelval IS NOT NULL THEN
                val := pixelval::integer;
                bitstring := bitstring || val::BIT(1);
            ELSE
                bitstring := bitstring || B'0'; -- or handle NULL as needed
            END IF;
        END LOOP;
    END LOOP;

    RETURN bitstring;
END;
$$
LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION bitstring_to_raster(
    bitstring BIT VARYING,
    raster_height INTEGER,
    raster_width INTEGER,
    srid INTEGER DEFAULT 0
)
RETURNS RASTER AS $$
DECLARE
    raster RASTER;
    pixel_type TEXT := '1BB'; -- 1-bit Boolean
    band_index INTEGER := 1;
    row INTEGER;
    col INTEGER;
    bit_value CHAR;
BEGIN
    -- Create an empty raster with the given width, height, and spatial reference system
    raster := ST_AddBand(
        ST_MakeEmptyRaster(
            raster_height,         -- Number of columns
            raster_width,          -- Number of rows
            0, 0,                  -- Upper-left X and Y (origin) of the raster
            1, 1,                  -- Pixel size X and Y (assuming square pixels)
            0, 0,                  -- X and Y skew
            srid                   -- Spatial Reference ID (SRID)
        ),
        band_index,
        pixel_type,
        0,
        0
    );

    -- Set the raster values from the bitstring
    FOR row IN 0..(raster_width - 1) LOOP
        FOR col IN 0..(raster_height - 1) LOOP
            bit_value := SUBSTRING(bitstring FROM row * raster_height + col + 1 FOR 1);
            IF bit_value = '1' THEN
                raster := ST_SetValue(raster, band_index, col + 1, row + 1, 1);
            ELSE
                raster := ST_SetValue(raster, band_index, col + 1, row + 1, 0);
            END IF;
        END LOOP;
    END LOOP;

    RETURN raster;
END;
$$ LANGUAGE plpgsql;


create table bitmask
(
    id         serial primary key,
    value      bit varying not null,
    height     int not null,
    width      int not null,
    boundary   geometry(Polygon) not null,
    created_at timestamp not null
);

create index ix_bitmask_id
    on bitmask (id);

ALTER TABLE annotation ADD bitmask_id integer references bitmask;

-- TODO convert rasters to bitmask and drop column

ALTER TABLE annotation DROP COLUMN raster;