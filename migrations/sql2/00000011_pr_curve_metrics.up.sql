-- Update 'pr_curve_iou_threshold' to 0.5 if it is NULL
UPDATE evaluation
SET parameters = jsonb_set(parameters, '{pr_curve_iou_threshold}', '0.5'::jsonb, false)
WHERE (
    NOT parameters ? 'pr_curve_iou_threshold'
    OR parameters->>'pr_curve_iou_threshold' IS NULL
);


CREATE OR REPLACE FUNCTION convert_pr_curve(input_json jsonb)
RETURNS jsonb LANGUAGE plpgsql AS $$
DECLARE
    label_value text;
    score_threshold text;
    metric_key text;
    metric_value jsonb;
    label_value_dict jsonb;
    score_threshold_dict jsonb;
    metric_dict jsonb;
BEGIN
    FOR label_value, label_value_dict IN SELECT * FROM jsonb_each(input_json)
    LOOP
        FOR score_threshold, score_threshold_dict IN SELECT * FROM jsonb_each(label_value_dict)
        LOOP
            FOR metric_key, metric_value IN SELECT * FROM jsonb_each(score_threshold_dict)
            LOOP
                IF jsonb_typeof(metric_value) = 'array' THEN
                    input_json = jsonb_set(input_json, ARRAY[label_value, score_threshold, metric_key], to_jsonb(jsonb_array_length(metric_value)), false);
                END IF;
            END LOOP;
        END LOOP;
    END LOOP;

    RETURN input_json;
END $$;


-- Convert 'PrecisionRecallCurve' metrics to the new schema format.
UPDATE metric
SET value = convert_pr_curve(value)
WHERE metric.type = 'PrecisionRecallCurve';
