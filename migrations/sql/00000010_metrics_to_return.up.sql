-- set 'metrics_to_return' based on 'task_type'
UPDATE evaluation
SET parameters = jsonb_set(
    parameters,
    '{metrics_to_return}',
    CASE
        WHEN parameters->>'task_type' = 'classification' THEN '["Accuracy","Precision","Recall","F1","ROCAUC"]'::jsonb
        WHEN parameters->>'task_type' = 'object-detection' THEN '["AP","AR","mAP","APAveragedOverIOUs","mAR","mAPAveragedOverIOUs"]'::jsonb
        WHEN parameters->>'task_type' = 'semantic-segmentation' THEN '["IOU", "mIOU"]'::jsonb
        ELSE '[]'::jsonb
    END,
    true
);

-- append 'PrecisionRecallCurve' to 'metrics_to_return' if 'compute_pr_curves' is true
UPDATE evaluation
SET parameters = jsonb_set(
    parameters,
    '{metrics_to_return}',
    CASE
        WHEN (parameters->>'compute_pr_curves')::boolean IS TRUE THEN
            COALESCE(parameters->'metrics_to_return', '[]'::jsonb) || '["PrecisionRecallCurve"]'::jsonb
        ELSE
            parameters->'metrics_to_return'
    END,
    true
)
WHERE evaluation.parameters ? 'compute_pr_curves';

-- Remove 'compute_pr_curves'
UPDATE evaluation
SET parameters = parameters - 'compute_pr_curves';