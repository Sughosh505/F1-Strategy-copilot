import os
import sys
import json
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

# Ensure workspace root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.snapshot import assemble_snapshot

try:
    # pyrefly: ignore [missing-import]
    from google import genai
    # pyrefly: ignore [missing-import]
    from google.genai import types
except ImportError:
    genai = None

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

class OrchestratorResponse(BaseModel):
    call: Literal["pit_now", "stay_out", "prep_for_sc_window"] = Field(
        description="The ranked strategic call."
    )
    rationale: str = Field(
        description="A 1-2 sentence explanation naming specific snapshot fields."
    )

def evaluate_strategy(year: int, race: str, lap: int, driver_code: str) -> dict:
    """
    Evaluates the strategy for a driver at a specific lap using the LLM orchestrator.
    """
    try:
        snapshot = assemble_snapshot(year, race, lap, driver_code)
    except Exception as e:
        return {
            "call": "stay_out",
            "rationale": f"Data error - Failed to assemble snapshot: {str(e)}"
        }

    return evaluate_snapshot(snapshot)

def evaluate_snapshot(snapshot: dict) -> dict:
    """
    Evaluates a specific JSON snapshot using the LLM orchestrator.
    """
    if genai is None:
        return {
            "call": "stay_out",
            "rationale": "API error - google-genai package not installed."
        }
        
    if not API_KEY:
        return {
            "call": "stay_out",
            "rationale": "API error - GEMINI_API_KEY not found in environment."
        }

    client = genai.Client(api_key=API_KEY)

    system_instruction = (
        "You are an expert F1 Race Strategist orchestrator. Your job is to make a single "
        "strategic call for the requested driver based STRICTLY on the provided JSON snapshot.\n\n"
        "CRITICAL RULES:\n"
        "1. GROUNDED REASONING ONLY: You may only use the numbers, flags, and text present in "
        "the provided JSON snapshot. Do NOT use outside general F1 knowledge, do not invent "
        "statistics, and do not hallucinate driver rivalries that are not flagged in `field_context`.\n"
        "2. SC PROBABILITY VS. ACTUAL SC:\n"
        "   - If `track_status` shows an active SC or VSC, this is a near-certain trigger to pit (cheap pit stop window).\n"
        "   - If `track_status` is Green, `sc_probability` is ONLY a minor timing nudge. You may ONLY recommend "
        "a pit stop based on SC probability if the driver's `pit_urgency` is ALREADY 'medium' or 'high'. "
        "DO NOT recommend pitting if tires are fresh (low pit_urgency) just because SC probability is elevated.\n"
        "3. TRACEABILITY: Your rationale MUST explicitly name the fields from the snapshot that drove "
        "your decision (e.g., \"pit_now selected because pit_urgency is high (0.85) and HAM poses an undercut_threat\").\n\n"
        "Output your response adhering to the requested JSON schema."
    )

    prompt = f"Here is the race snapshot:\n```json\n{json.dumps(snapshot, indent=2)}\n```\nMake your strategic call."

    try:
        response = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=OrchestratorResponse,
                temperature=0.0,
            ),
        )
        
        return json.loads(response.text)
    except Exception as e:
        print(f"[ORCHESTRATOR ERROR] {e}")
        return {
            "call": "stay_out",
            "rationale": "API error or timeout - defaulting to conservative stay_out call."
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--race", type=str, default="Dutch Grand Prix")
    parser.add_argument("--lap", type=int, default=40)
    parser.add_argument("--driver", type=str, default="VER")
    args = parser.parse_args()

    result = evaluate_strategy(args.year, args.race, args.lap, args.driver)
    print("\n### Orchestrator Recommendation ###")
    print(json.dumps(result, indent=2))
