-- Remove 'pr_curve_max_examples'
UPDATE evaluation
SET parameters = parameters - 'pr_curve_max_examples';
