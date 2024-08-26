CREATE OR REPLACE FUNCTION raster_to_bitstring(input_raster raster)
RETURNS BIT VARYING AS
$$
DECLARE
    height INTEGER;
    width INTEGER;
    bitstring BIT VARYING := B'';
    val INTEGER;
    pixelval INTEGER;
    colx INTEGER;
    rowy INTEGER;
BEGIN
    SELECT ST_Width(input_raster), ST_Height(input_raster) INTO width, height;
    FOR rowy IN 1..height LOOP
        FOR colx IN 1..width LOOP
            pixelval := ST_Value(input_raster, colx, rowy);
            IF pixelval IS NOT NULL THEN
                val := pixelval::integer;
                bitstring := bitstring || val::BIT(1);
            ELSE
                bitstring := bitstring || B'0';
            END IF;
        END LOOP;
    END LOOP;

    RETURN bitstring;
END;
$$
LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION bitstring_to_raster(
    bitstring BIT VARYING,
    height INTEGER,
    width INTEGER,
    srid INTEGER DEFAULT 0
)
RETURNS RASTER AS $$
DECLARE
    raster RASTER;
    pixel_type TEXT := '1BB';
    colx INTEGER;
    rowy INTEGER;
    bit_value CHAR;
BEGIN
    raster := ST_AddBand(
        ST_MakeEmptyRaster(
            width,          -- Number of rows
            height,         -- Number of columns
            0, 0,           -- Upper-left (x,y)
            1, 1,           -- Pixel size (x,y)
            0, 0,           -- Skew (x,y)
            srid            -- Spatial Reference ID (SRID)
        ),
        1,              -- Band Index
        pixel_type,     -- Pixel Type
        0,              -- Precision
        0               -- NoData Value
    );
    FOR rowy IN 1..height LOOP
        FOR colx IN 1..width LOOP
            -- Note: bitstring index starts from 1
            bit_value := SUBSTRING(bitstring FROM ((rowy - 1) * width) + colx FOR 1);
            IF bit_value = '1' THEN
                raster := ST_SetValue(raster, 1, colx, rowy, 1);
            ELSE
                raster := ST_SetValue(raster, 1, colx, rowy, 0);
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
    boundary   geometry(Polygon),
    created_at timestamp not null
);

create index ix_bitmask_id
    on bitmask (id);

ALTER TABLE annotation ADD bitmask_id integer references bitmask;

-- TODO convert rasters to bitmask and drop column

ALTER TABLE annotation DROP COLUMN raster;