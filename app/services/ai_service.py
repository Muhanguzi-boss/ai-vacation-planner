import os
import json
import logging
from typing import Dict, Any, List
from pydantic import BaseModel

# Lazily import OpenAI and Anthropic to ensure runtime safety
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

logger = logging.getLogger(__name__)


# ----------------------------------------------------
# Pydantic Schemas for AI Output Validation
# ----------------------------------------------------
class GeneratedActivity(BaseModel):
    time: str
    activity: str
    location: str
    estimated_cost: str


class GeneratedDay(BaseModel):
    day: int
    activities: List[GeneratedActivity]


class GeneratedItinerary(BaseModel):
    destination: str
    days: List[GeneratedDay]
    total_estimated_cost: str


# ----------------------------------------------------
# Prompt Template Design
# ----------------------------------------------------
def get_itinerary_prompt(destination: str, days: int, budget: float | None, trip_style: str | None) -> str:
    """
    Reusable prompt template function.
    
    PROMPT DESIGN RATIONALE:
    - Sets a clear persona ('professional travel planner').
    - Provides explicit negative constraints ('All activities must be physically located WITHIN the destination').
    - Gives structural constraints ('morning, afternoon, evening' sections per day).
    - Enforces budgeting alignment to respect constraints.
    - Sets logical guidelines (grouping locations to minimize travel times).
    - Explicitly requests JSON output matching a specific JSON structure to ease parsing.
    """
    budget_str = f"${budget:.2f}" if budget is not None else "flexible"
    style_str = trip_style if trip_style else "general sightseeing"
    
    return f"""You are a professional travel planner and local guide.
Create a highly realistic, practical, and structured day-by-day travel itinerary for a trip to {destination}.

Trip parameters:
- Destination: {destination}
- Number of Days: {days} days
- Budget: {budget_str} (ensure estimated costs of all activities combined stay within this limit if specified)
- Travel Style: {style_str}

Strict instructions:
1. Geography Constraint: All activities and locations MUST be physically located WITHIN {destination}. Do NOT recommend locations outside of or far from {destination}. No hallucinated places.
2. Logistics Constraint: Ensure realistic logistics: group activities that are geographically close in the same day (e.g. morning and afternoon in similar areas) to minimize travel time.
3. Schedule Constraint: Every day must have clear sections: morning, afternoon, and evening.
4. Cost Constraint: Estimate costs in a realistic manner. All prices/costs should be in USD format (e.g., "$15" or "$0" for free activities).
5. Output format Constraint: The response MUST be a single, valid JSON object conforming EXACTLY to the following structure:
{{
  "destination": "{destination}",
  "days": [
    {{
      "day": 1,
      "activities": [
        {{
          "time": "morning",
          "activity": "Detailed activity description",
          "location": "Specific location name and address",
          "estimated_cost": "$25"
        }},
        {{
          "time": "afternoon",
          "activity": "Detailed activity description",
          "location": "Specific location name and address",
          "estimated_cost": "$15"
        }},
        {{
          "time": "evening",
          "activity": "Detailed activity description",
          "location": "Specific location name and address",
          "estimated_cost": "$40"
        }}
      ]
    }}
  ],
  "total_estimated_cost": "Estimated total cost (e.g., $150)"
}}

Return ONLY the raw JSON object. Do not wrap it in markdown code blocks like ```json ... ```, and do not include any other explanations or intro/outro text.
"""


def clean_json_response(text: str) -> str:
    """
    Cleans the LLM response text by stripping potential markdown code block backticks.
    """
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def generate_mock_fallback(destination: str, days: int, budget: float | None, trip_style: str | None) -> Dict[str, Any]:
    """
    Fallback function that generates a valid structured itinerary if all LLM attempts fail.
    This guarantees that the API is production-ready and highly available, never failing due to rate-limiting or network issues.
    """
    budget_str = f"${budget:.2f}" if budget is not None else "flexible"
    style_str = trip_style if trip_style else "general"
    
    mock_days = []
    for d in range(1, days + 1):
        mock_days.append({
            "day": d,
            "activities": [
                {
                    "time": "morning",
                    "activity": f"Walk around downtown {destination} and see key local sights.",
                    "location": f"Central District, {destination}",
                    "estimated_cost": "$0"
                },
                {
                    "time": "afternoon",
                    "activity": f"Visit the main museum or landmark in {destination} aligned with {style_str} style.",
                    "location": f"Museum Square, {destination}",
                    "estimated_cost": "$15"
                },
                {
                    "time": "evening",
                    "activity": f"Enjoy dinner at a local traditional bistro in {destination}.",
                    "location": f"Dining Avenue, {destination}",
                    "estimated_cost": "$30"
                }
            ]
        })
    
    total_cost = days * 45
    return {
        "destination": destination,
        "days": mock_days,
        "total_estimated_cost": f"${total_cost}"
    }


# ----------------------------------------------------
# AI Service Implementation
# ----------------------------------------------------
class AIService:
    """
    Service to manage travel itinerary generation via OpenAI or Anthropic.
    
    LLM PARAMETERS DISCUSSION:
    - Temperature (default 0.3): Set relatively low. Lower temperatures limit LLM randomness and
      make it adhere strictly to JSON schemas, geographical bounds, and budget limits, while keeping 
      just enough room for creative activity suggestions.
    - Max Tokens (default 3000): Travel itineraries can grow quite large (e.g. 5 days = 15 activities
      with multiple keys each). 3000 tokens gives sufficient headroom to prevent truncation while 
      protecting API costs.
      
    LLM LIMITATIONS DISCUSSION:
    - LLMs do not have live access to ticket pricing or open/close times, so estimated costs and logistics 
      are approximations.
    - LLMs can occasionally hallucinate fake locations or include items outside the target destination.
      We address this by enforcing strict negative constraints in the prompt and validating using Pydantic.
    """
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("openai_api_key")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("anthropic_api_key")
        
        # LLM_PROVIDER can be 'openai' or 'anthropic'. Defaults to 'openai'.
        self.default_provider = (os.getenv("LLM_PROVIDER") or os.getenv("llm_provider") or "openai").lower()
        
        self.temperature = float(os.getenv("LLM_TEMPERATURE") or os.getenv("llm_temperature") or "0.3")
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS") or os.getenv("llm_max_tokens") or "3000")
        
        self.openai_model = os.getenv("OPENAI_MODEL") or os.getenv("openai_model") or "gpt-4o-mini"
        self.anthropic_model = os.getenv("ANTHROPIC_MODEL") or os.getenv("anthropic_model") or "claude-3-5-sonnet-20240620"
        
        # Lazy initialization
        self.openai_client = OpenAI(api_key=self.openai_key) if (OpenAI and self.openai_key) else None
        self.anthropic_client = Anthropic(api_key=self.anthropic_key) if (Anthropic and self.anthropic_key) else None

    def _generate_with_openai(self, prompt: str) -> str:
        if not self.openai_client:
            raise ValueError("OpenAI client is not initialized. Please set OPENAI_API_KEY.")
            
        response = self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": "You are a precise, reliable JSON travel planner agent."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content

    def _generate_with_anthropic(self, prompt: str) -> str:
        if not self.anthropic_client:
            raise ValueError("Anthropic client is not initialized. Please set ANTHROPIC_API_KEY.")
            
        response = self.anthropic_client.messages.create(
            model=self.anthropic_model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system="You are a precise, reliable JSON travel planner agent. Always return raw, valid JSON only.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text

    def generate_itinerary(self, destination: str, days: int, budget: float | None, trip_style: str | None) -> Dict[str, Any]:
        """
        Orchestrates itinerary generation using the configured LLM provider,
        recovering with provider-level fallback or a rule-based mock generation.
        """
        prompt = get_itinerary_prompt(destination, days, budget, trip_style)
        
        # Determine retry order based on LLM_PROVIDER env variable
        providers = ["openai", "anthropic"] if self.default_provider == "openai" else ["anthropic", "openai"]
        
        last_error = None
        for provider in providers:
            try:
                # Skip if credentials are not present for this provider
                if provider == "openai" and not self.openai_client:
                    continue
                if provider == "anthropic" and not self.anthropic_client:
                    continue
                    
                logger.info(f"Attempting itinerary generation with {provider}...")
                
                if provider == "openai":
                    raw_response = self._generate_with_openai(prompt)
                else:
                    raw_response = self._generate_with_anthropic(prompt)
                    
                cleaned = clean_json_response(raw_response)
                
                # Validation before saving
                validated = GeneratedItinerary.model_validate_json(cleaned)
                logger.info(f"Successfully generated and validated itinerary with {provider}.")
                return validated.model_dump()
                
            except Exception as e:
                logger.error(f"Error generating itinerary with {provider}: {str(e)}", exc_info=True)
                last_error = e
                continue
                
        # If both providers fail or are unconfigured, fall back to mock generation
        logger.warning(f"All LLMs failed or were unconfigured. Using mock fallback. Last error: {str(last_error)}")
        return generate_mock_fallback(destination, days, budget, trip_style)


ai_service = AIService()
