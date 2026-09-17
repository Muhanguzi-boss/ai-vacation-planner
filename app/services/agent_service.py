from __future__ import annotations

import os
from typing import Any, Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.services.ai_service import GeneratedItinerary, ai_service
from app.tools import travel_knowledge_search_tool, weather_tool

MAX_TOOL_ITERATIONS = 3


class TravelPlanningState(TypedDict):
    user_request: str
    messages: Annotated[list[AnyMessage], add_messages]
    itinerary: GeneratedItinerary | None
    tool_iterations: int


class OrchestrationError(RuntimeError):
    """Raised when the travel-planning graph cannot complete."""


class TravelPlanningOrchestrator:
    def __init__(
        self,
        model: Any | None = None,
        tools: list[BaseTool] | None = None,
    ) -> None:
        self.tools = tools or [travel_knowledge_search_tool, weather_tool]
        self.model = model or self._create_anthropic_model()
        self.graph = self._build_graph()

    @staticmethod
    def _create_anthropic_model() -> ChatAnthropic:
        api_key = ai_service.anthropic_key
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for travel orchestration")

        return ChatAnthropic(
            model_name=ai_service.anthropic_model,
            api_key=api_key,
            temperature=ai_service.temperature,
            max_tokens_to_sample=ai_service.max_tokens,
        )

    def _build_graph(self):
        model_with_tools = self.model.bind_tools(self.tools)
        structured_model = self.model.with_structured_output(GeneratedItinerary)
        tool_node = ToolNode(self.tools, handle_tool_errors=False)

        def plan(state: TravelPlanningState) -> dict[str, Any]:
            response = model_with_tools.invoke(state["messages"])
            tool_iterations = state["tool_iterations"]
            if getattr(response, "tool_calls", None):
                tool_iterations += 1
                if tool_iterations > MAX_TOOL_ITERATIONS:
                    raise OrchestrationError(
                        f"Travel orchestration exceeded the {MAX_TOOL_ITERATIONS}-iteration tool limit"
                    )
            return {"messages": [response], "tool_iterations": tool_iterations}

        def route_after_plan(state: TravelPlanningState) -> str:
            last_message = state["messages"][-1]
            if (
                getattr(last_message, "tool_calls", None)
                and state["tool_iterations"] > MAX_TOOL_ITERATIONS
            ):
                raise OrchestrationError(
                    f"Travel orchestration exceeded the {MAX_TOOL_ITERATIONS}-iteration tool limit"
                )
            return "tools" if getattr(last_message, "tool_calls", None) else "finalize"

        def finalize(state: TravelPlanningState) -> dict[str, GeneratedItinerary]:
            response = structured_model.invoke(
                [
                    *state["messages"],
                    HumanMessage(
                        content=(
                            "Create the final itinerary for the user's request. "
                            "Use the available tool results as reference context. "
                            "Return only the structured itinerary."
                        )
                    ),
                ]
            )
            itinerary = (
                response
                if isinstance(response, GeneratedItinerary)
                else GeneratedItinerary.model_validate(response)
            )
            return {"itinerary": itinerary}

        builder = StateGraph(TravelPlanningState)
        builder.add_node("plan", plan)
        builder.add_node("tools", tool_node)
        builder.add_node("finalize", finalize)
        builder.add_edge(START, "plan")
        builder.add_conditional_edges(
            "plan",
            route_after_plan,
            {"tools": "tools", "finalize": "finalize"},
        )
        builder.add_edge("tools", "plan")
        builder.add_edge("finalize", END)
        return builder.compile()

    def invoke(self, user_request: str) -> GeneratedItinerary:
        if not isinstance(user_request, str) or not user_request.strip():
            raise ValueError("user_request must be a non-empty string")

        initial_state: TravelPlanningState = {
            "user_request": user_request.strip(),
            "messages": [
                SystemMessage(
                    content=(
                        "You are a travel-planning assistant. Use travel knowledge "
                        "and weather tools only when they are relevant to the request."
                    )
                ),
                HumanMessage(content=user_request.strip()),
            ],
            "itinerary": None,
            "tool_iterations": 0,
        }
        try:
            result = self.graph.invoke(initial_state)
        except Exception as exc:
            raise OrchestrationError(f"Travel orchestration failed: {exc}") from exc

        itinerary = result.get("itinerary")
        if not isinstance(itinerary, GeneratedItinerary):
            raise OrchestrationError("Travel orchestration returned no valid itinerary")
        return itinerary
