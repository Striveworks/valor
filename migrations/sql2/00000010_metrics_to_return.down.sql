-- Add 'compute_pr_curves' back to 'parameters' based on whether 'metrics_to_return' contains 'PrecisionRecallCurve'
UPDATE evaluation
SET parameters = jsonb_set(
    parameters,
    '{compute_pr_curves}',
    (parameters->'metrics_to_return' ? 'PrecisionRecallCurve')::boolean,
    true
);

-- Remove 'metrics_to_return'
UPDATE evaluation
SET parameters = parameters - 'metrics_to_return';
