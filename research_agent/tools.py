import os
import requests
from dotenv import load_dotenv
from google.adk.tools import ToolContext

load_dotenv()

SAPLING_API_KEY = os.getenv("SAPLING_API_KEY")
SAPLING_URL = "https://api.sapling.ai/api/v1/aidetect"


def check_sapling_ai_score(text: str, tool_context: ToolContext) -> dict:
    """Sends text to Sapling AI Detector API and returns the AI probability percentage.

    Stores the result in `current_score` every call, and additionally stores it
    in `initial_score` the first time it's ever called, so the baseline isn't
    overwritten by later loop iterations. Also caches the very first text it
    ever sees as `original_text`, so later agents can reference the original
    input via state instead of relying on full conversation history.
    """
    if not SAPLING_API_KEY:
        return {"status": "error", "message": "SAPLING_API_KEY is not set in environment."}

    if "original_text" not in tool_context.state:
        tool_context.state["original_text"] = text

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SAPLING_API_KEY}"
    }
    payload = {"text": text}

    try:
        response = requests.post(SAPLING_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Sapling score is returned as a float (0.0 to 1.0)
        score_percent = round(data.get("score", 0.0) * 100, 2)

        # Always update the running score
        tool_context.state["current_score"] = score_percent

        # Only set initial_score once, so later iterations don't clobber it
        if "initial_score" not in tool_context.state:
            tool_context.state["initial_score"] = score_percent

        return {
            "status": "success",
            "ai_score_percent": score_percent
        }
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Sapling API request failed: {str(e)}"}
    except ValueError as e:
        # response.json() failed to parse
        return {"status": "error", "message": f"Could not parse Sapling response: {str(e)}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def load_prompt_guidelines(tool_context: ToolContext, filename: str = "rewrite_rules.md") -> dict:
    """Reads rewriting rules from the prompts/ directory and caches them in
    session state under `guidelines`, so this only needs to be called ONCE
    per run (e.g. by the baseline agent) rather than once per loop iteration.
    Later agents should read `{guidelines}` directly from state via an
    instruction placeholder instead of calling this tool again.
    """
    if "guidelines" in tool_context.state:
        return {"status": "success", "guidelines": tool_context.state["guidelines"], "cached": True}

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_dir, "prompts", filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        tool_context.state["guidelines"] = content
        return {"status": "success", "guidelines": content, "cached": False}
    except Exception as e:
        return {"status": "error", "message": f"Could not load guidelines: {str(e)}"}


def exit_loop(tool_context: ToolContext) -> dict:
    """Signals ADK LoopAgent to break the cycle immediately when AI score <= 20%."""
    tool_context.actions.escalate = True
    tool_context.state["target_reached"] = True
    return {"status": "exit_signal_sent", "message": "Target score reached (<=20%). Halting loop."}