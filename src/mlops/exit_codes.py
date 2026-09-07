"""Exit-code contract between scripts/monitor_drift.py and the agent runner.

Before issue #71 the script returned 1 for *both* "drift detected" and "the
analysis raised", and `ScriptResult.success` is `exit_code == 0`. So the
`drift_monitoring` workflow could not tell a result from a failure, and fell
back to scraping the human-readable report out of stdout -- where a check that
never ran left no line to scrape, and absence read as healthy. That is how a
drift check could fail on 223 of 232 runs, and report
"No action needed - all classifiers healthy" on 219 of those failures.

So the script has to say which of three things happened:

    EXIT_NO_DRIFT         the check ran and found no drift
    EXIT_DRIFT_DETECTED   the check ran and found drift -- a *result*, not an
                          error, which is why it is not retried
    EXIT_INDETERMINATE    the check produced no verdict at all: it raised, or
                          `DriftReport.indeterminate` was set because there was
                          nothing to compare. Never reported as healthy.

This deliberately mirrors `src/labeling/exit_codes.py` rather than importing
it. The two contracts share the integers 0/1/2 and nothing else: labeling's 2
means "partially done, do not re-run me", drift's means "I could not tell".
Each contract stays self-describing, so a reader of either module sees the
whole of it.
"""

EXIT_NO_DRIFT = 0
EXIT_DRIFT_DETECTED = 1
EXIT_INDETERMINATE = 2

# Exit codes the agent runner must not retry.
#
# Drift detection is a determinate result: re-running the same analysis over the
# same window returns the same answer, so retrying only delays the alert. Before
# #71 no non-retryable set was passed at all, so a genuine drift detection was
# retried `AGENT_MAX_RETRIES` times with exponential backoff before being
# reported.
#
# EXIT_INDETERMINATE is deliberately NOT in this set: it covers transient causes
# (the database is briefly unreachable) as well as durable ones (the reference
# dataset is stale). Retrying heals the first and merely spends the attempts on
# the second, which then reports `unknown` correctly.
NON_RETRYABLE_EXIT_CODES = frozenset({EXIT_DRIFT_DETECTED})
