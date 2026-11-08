"""Opt-in online itinerary evaluation; intentionally not collected by `pytest tests`."""

import json
import os
from pathlib import Path

import pytest

from app.services.ai_service import GeneratedItinerary
from app.services.agent_service import TravelPlanningOrchestrator


DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "evaluation"
    / "golden-itineraries-v1.json"
)
DATASET = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
MINIMUM_SCORE = 0.7


def _anthropic_model(api_key: str, model_name: str):
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=model_name,
        api_key=api_key,
        temperature=0,
        max_tokens=3000,
    )


def _deepeval_types():
    pytest.importorskip("deepeval", reason="Install requirements-evaluation.txt")
    from deepeval import assert_test
    from deepeval.metrics import GEval
    from deepeval.models.base_model import DeepEvalBaseLLM
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    return assert_test, GEval, DeepEvalBaseLLM, LLMTestCase, LLMTestCaseParams


def _anthropic_judge(api_key: str, model_name: str, base_llm_type):
    class AnthropicJudge(base_llm_type):
        def __init__(self):
            self.model = _anthropic_model(api_key, model_name)

        def load_model(self):
            return self.model

        def generate(self, prompt: str) -> str:
            response = self.model.invoke(prompt)
            return response.content if isinstance(response.content, str) else str(response.content)

        async def a_generate(self, prompt: str) -> str:
            response = await self.model.ainvoke(prompt)
            return response.content if isinstance(response.content, str) else str(response.content)

        def get_model_name(self) -> str:
            return f"Anthropic {model_name}"

    return AnthropicJudge()


def _validate_generated_output(output: str, case: dict) -> GeneratedItinerary:
    itinerary = GeneratedItinerary.model_validate_json(output)
    if itinerary.destination.casefold().strip() != case["destination"].casefold().strip():
        raise AssertionError("Generated itinerary destination did not match the golden case.")
    if len(itinerary.days) != case["days"]:
        raise AssertionError("Generated itinerary duration did not match the golden case.")
    for day in itinerary.days:
        times = {activity.time.casefold() for activity in day.activities}
        if len(day.activities) < 3 or not {"morning", "afternoon", "evening"} <= times:
            raise AssertionError("Each day must include at least three activities across morning, afternoon, and evening.")
    return itinerary


def _evaluation_metrics(GEval, LLMTestCaseParams, judge):
    common_params = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ]
    return [
        GEval(
            name="Itinerary relevance",
            criteria=(
                "Assess whether the itinerary directly serves the destination, duration, "
                "budget, and travel preferences in the request. Award 1 only when it is "
                "clearly relevant and does not substitute an unrelated trip."
            ),
            evaluation_params=common_params,
            threshold=MINIMUM_SCORE,
            model=judge,
            async_mode=False,
        ),
        GEval(
            name="Itinerary completeness",
            criteria=(
                "Assess whether every requested trip day has useful, specific activities "
                "for morning, afternoon, and evening, with locations and estimated costs. "
                "Penalize missing days, empty plans, and materially incomplete schedules."
            ),
            evaluation_params=common_params,
            threshold=MINIMUM_SCORE,
            model=judge,
            async_mode=False,
        ),
        GEval(
            name="Itinerary consistency",
            criteria=(
                "Assess internal consistency: destination and day numbering agree, nearby "
                "activities are sensibly grouped, costs and total are plausible together, "
                "and claims do not conflict with the request or each other."
            ),
            evaluation_params=common_params,
            threshold=MINIMUM_SCORE,
            model=judge,
            async_mode=False,
        ),
        GEval(
            name="Output structure adherence",
            criteria=(
                "Assess whether the output is a valid itinerary matching the required "
                "GeneratedItinerary structure: destination, a days list, sequential day "
                "numbers, activities with time/activity/location/estimated_cost, and "
                "total_estimated_cost. Penalize omitted or malformed fields."
            ),
            evaluation_params=common_params,
            threshold=MINIMUM_SCORE,
            model=judge,
            async_mode=False,
        ),
    ]


@pytest.mark.parametrize("case", DATASET["cases"], ids=lambda case: case["id"])
def test_golden_itinerary_quality(case):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("Set ANTHROPIC_API_KEY to run paid, online DeepEval evaluations.")

    model_name = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
    assert_test, GEval, DeepEvalBaseLLM, LLMTestCase, LLMTestCaseParams = _deepeval_types()

    request = (
        f"{case['request']} Destination: {case['destination']}. "
        f"Duration: {case['days']} days. "
        f"Travel style: {case['travel_style']}."
    )
    budget = "flexible" if case["budget"] is None else f"${case['budget']}"
    request += f" Budget: {budget}."
    generated = TravelPlanningOrchestrator(
        model=_anthropic_model(api_key, model_name),
    ).invoke(
        request,
        expected_destination=case["destination"],
        expected_days=case["days"],
    )
    actual_output = generated.model_dump_json()
    _validate_generated_output(actual_output, case)

    judge = _anthropic_judge(api_key, model_name, DeepEvalBaseLLM)
    test_case = LLMTestCase(
        input=request,
        actual_output=actual_output,
        expected_output=json.dumps(case["expected_output"], ensure_ascii=False),
    )
    assert_test(test_case, _evaluation_metrics(GEval, LLMTestCaseParams, judge))
