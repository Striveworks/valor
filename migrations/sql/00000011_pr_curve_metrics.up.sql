-- Update 'pr_curve_iou_threshold' to 0.5 if it is NULL
UPDATE evaluation
SET parameters = jsonb_set(parameters,'{pr_curve_iou_threshold}', '0.5'::jsonb, false)
WHERE parameters->'pr_curve_iou_threshold' IS NULL;

CREATE OR REPLACE FUNCTION convert_pr_curve(jsonb)
RETURNS jsonb LANGUAGE plpgsql AS $$
DECLARE
    label_value text;
    score_threshold text;
    metric_key text;
    metric_value jsonb;
    label_value_dict jsonb;
    score_threshold_dict jsonb;
    metric_dict jsonb;
    result jsonb := '{}'::jsonb;
BEGIN
    FOR label_value, label_value_dict IN SELECT * FROM jsonb_each($1)
    LOOP
        FOR score_threshold, score_threshold_dict IN SELECT * FROM jsonb_each(label_value_dict)
        LOOP
            metric_dict := '{}'::jsonb;
            FOR metric_key, metric_value IN SELECT * FROM jsonb_each(score_threshold_dict)
            LOOP
                IF jsonb_typeof(metric_value) != 'array' THEN
                    metric_dict := jsonb_set(metric_dict, '{' || metric_key || '}', metric_value);
                END IF;
            END LOOP;
            result := jsonb_set(result, '{' || label_value || ',' || score_threshold || '}', metric_dict);
        END LOOP;
    END LOOP;
    RETURN result;
END $$;

-- Convert 'PrecisionRecall' metrics to the new schema format.
UPDATE metric
SET value = convert_pr_curve(value)
WHERE metric.type = 'PrecisionRecallCurve';
