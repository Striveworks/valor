# %%
import csv
import json
import sys

# bump max csv size
csv.field_size_limit(sys.maxsize)

# Step 1: Use pgadmin to fetch data in json format
sql_script = """
with annotations AS (
    select a.id as annotation_id,
        a.datum_id,
        b.meta as datum_metadata,
        ST_AsGeoJSON(ST_Envelope(ST_MinConvexHull(raster))) as "raster",
        ST_AsGeoJSON(ST_Envelope(polygon)) as "polygon",
        ST_AsGeoJSON(box) as "box",
        is_instance
    from annotation a
    inner join datum b on a.datum_id=b.id
),
annotations_per_groundtruth as (
    select datum_id,
        datum_metadata,
        b.id as groundtruth_id,
        a.annotation_id,
        raster,
        polygon,
        box,
        is_instance,
        array_agg(jsonb_build_object('key', c.key, 'value', c.value)) as labels
    from annotations a
    inner join groundtruth b on a.annotation_id=b.annotation_id
    inner join label c on b.label_id=c.id
    group by datum_id, datum_metadata, a.annotation_id, b.id, raster, polygon, box, is_instance
),
annotations_per_prediction as (
    select datum_id,
        datum_metadata,
        b.id as prediction_id,
        a.annotation_id,
        raster,
        polygon,
        box,
        is_instance,
        array_agg(jsonb_build_object('key', c.key, 'value', c.value, 'score', b.score)) as labels
    from annotations a
    inner join prediction b on a.annotation_id=b.annotation_id
    inner join label c on b.label_id=c.id
    group by datum_id, datum_metadata, a.annotation_id, b.id, raster, polygon, box, is_instance
),
gts_per_datum as (
    select datum_id,
        datum_metadata,
        array_agg(jsonb_build_object('raster', raster, 'polygon', polygon, 'box', box, 'labels', labels, 'is_instance', is_instance)) as anns
    from annotations_per_groundtruth
    group by datum_id, datum_metadata
),
pds_per_datum as (
    select datum_id,
        datum_metadata,
        array_agg(jsonb_build_object('raster', raster, 'polygon', polygon, 'box', box, 'labels', labels, 'is_instance', is_instance)) as anns
    from annotations_per_prediction
    group by datum_id, datum_metadata
),
final_ as (
    select a.datum_id,
        a.datum_metadata,
        a.anns as groundtruth_annotations,
        b.anns as prediction_annotations
    from gts_per_datum a
    inner join pds_per_datum b on a.datum_id=b.datum_id
    order by a.datum_id desc
)
SELECT row_to_json(final_)
FROM final_;
"""

# Step 2: Click download to download to csv (pgadmin only allows you to download CSVs)


# Step 3: Convert the csv file to json using the code below
def convert_csv_to_json(csv_path, json_path):
    data = {}
    with open(csv_path, encoding="utf-8") as csvf:
        csvReader = csv.DictReader(csvf)
        for rows in csvReader:
            json_data = json.loads(rows["row_to_json"])
            datum_id = json_data["datum_id"]
            data[datum_id] = json_data
    print(data)
    with open(json_path, "w", encoding="utf-8") as jsonf:
        jsonf.write(json.dumps(data, indent=4))


convert_csv_to_json("data.csv", "data.json")

# %%
