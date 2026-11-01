-- Keep the run outcome comments aligned with quota-stop counts.
COMMENT ON COLUMN labeling_runs.articles_failed IS
    'Distinct articles affected by errors in the run, not API request attempts. '
    'A quota stop includes pending articles that would reach the LLM but were '
    'left uncalled after the first quota response.';

COMMENT ON COLUMN labeling_runs.articles_left_pending IS
    'Final count of articles still pending after the run, including articles '
    'left unprocessed when a quota error stopped the batch. This may overlap '
    'articles_failed.';
