def get_weather(destination: str) -> dict:
    """Return simple weather context for the supplied destination.

    This keeps the AI service decoupled from external APIs while still providing
    useful context for prompt construction.
    """
    destination_key = destination.lower()
    if "paris" in destination_key:
        return {"summary": "mostly sunny, average 24°C, light breeze"}
    if "rome" in destination_key:
        return {"summary": "warm and clear, average 29°C"}
    if "tokyo" in destination_key:
        return {"summary": "mild with occasional rain, average 22°C"}
    return {"summary": "pleasant weather with a mix of sun and clouds"}
