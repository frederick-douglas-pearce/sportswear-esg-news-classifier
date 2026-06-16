"""LLM-based reflection on completed experiments.

Follows the LabelingAnalyzer pattern from src/agent/llm.py.
Takes a completed ExperimentEntry, asks Claude to fill reflection fields.
Never raises — returns reflection with error in summary on failure.
"""

import json
import logging
import re
import time

from anthropic import Anthropic, RateLimitError

from .models import ExperimentEntry, ExperimentReflection

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"

REFLECTION_SYSTEM_PROMPT = """You are an ML experiment analyst reviewing a completed model training experiment.

Your task is to reflect on the experiment and provide:
1. A concise summary of what happened and why
2. Whether the hypothesis was confirmed, refuted, or inconclusive
3. Any surprising findings
4. Recommended next steps
5. Your confidence level in the conclusions

Respond with a JSON object only — no markdown formatting or explanation outside the JSON."""

REFLECTION_USER_PROMPT = """Analyze this completed ML experiment:

## Classifier
{classifier}

## Pre-experiment State
- Production version: {production_version}
- Production metrics: {production_metrics}
- Training data: {training_data}

## Action
- Type: {action_type}
- Summary: {action_summary}
- Changes: {changes}

## Observed Results
- Metrics: {metrics}
- Delta from production: {delta}

## Promotion Outcome
- Promoted: {promoted}
- Promotion reason: {promotion_reason}
- Target met: {target_met}

Respond with a JSON object in this format:
{{
    "summary": "2-3 sentence summary of the experiment",
    "hypothesis_result": "confirmed|refuted|inconclusive",
    "hypothesis_evidence": "brief evidence supporting the result assessment",
    "surprises": ["list of unexpected findings, if any"],
    "next_steps": ["list of recommended next actions"],
    "confidence": "low|medium|high",
    "confidence_notes": "brief explanation of confidence level"
}}"""


class ExperimentReflector:
    """Generates LLM-based reflections on completed experiments."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = Anthropic(api_key=api_key)

    def reflect(self, experiment: ExperimentEntry) -> ExperimentReflection:
        """Generate a reflection for a completed experiment.

        Never raises — returns reflection with error in summary on failure.
        """
        try:
            user_prompt = REFLECTION_USER_PROMPT.format(
                classifier=experiment.classifier,
                production_version=experiment.state.production_version or "none",
                production_metrics=json.dumps(experiment.state.production_metrics),
                training_data=json.dumps(experiment.state.training_data),
                action_type=experiment.action.type,
                action_summary=experiment.action.summary,
                changes=json.dumps(
                    [c.model_dump() for c in experiment.action.changes]
                )
                if experiment.action.changes
                else "none",
                metrics=json.dumps(experiment.observation.metrics),
                delta=json.dumps(experiment.observation.delta),
                promoted=experiment.reward.promoted,
                promotion_reason=experiment.reward.promotion_reason,
                target_met=experiment.reward.target_met,
            )

            return self._call_api(user_prompt)

        except Exception as e:
            logger.error(f"Reflection failed: {e}")
            return ExperimentReflection(
                summary=f"Reflection failed: {e}",
                confidence="low",
            )

    def _call_api(self, user_prompt: str) -> ExperimentReflection:
        """Call Claude API with retry logic."""
        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    temperature=0.0,
                    system=REFLECTION_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}],
                )

                response_text = response.content[0].text
                data = self._parse_json(response_text)

                return ExperimentReflection(
                    summary=data.get("summary", ""),
                    hypothesis_result=data.get("hypothesis_result", "inconclusive"),
                    hypothesis_evidence=data.get("hypothesis_evidence", ""),
                    surprises=data.get("surprises", []),
                    next_steps=data.get("next_steps", []),
                    confidence=data.get("confidence", "medium"),
                    confidence_notes=data.get("confidence_notes", ""),
                )

            except RateLimitError as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    logger.warning(f"Rate limited, retrying in {delay}s: {e}")
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Rate limit exceeded after {self.max_retries} attempts"
                    )
                    return ExperimentReflection(
                        summary=f"Reflection failed: rate limit exceeded",
                        confidence="low",
                    )

            except Exception as e:
                logger.error(f"Reflection API call failed: {e}")
                return ExperimentReflection(
                    summary=f"Reflection failed: {e}",
                    confidence="low",
                )

        return ExperimentReflection(
            summary="Reflection failed: max retries exceeded",
            confidence="low",
        )

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Parse JSON from response, handling markdown code blocks."""
        # Try markdown code block first
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if json_match:
            return json.loads(json_match.group(1).strip())

        # Try raw JSON object
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            return json.loads(json_match.group(0))

        return {}
