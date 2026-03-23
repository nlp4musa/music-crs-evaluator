
from __future__ import annotations
import json
import os
from typing import Optional
from google import genai

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "eval_prompt.txt")


class Gemini_LLM_Client:
    """Gemini text generation client."""
    def __init__(
        self,
        model_name: str = "gemini-3.0-flash",
        max_new_tokens: int = 8192,
        api_key: Optional[str] = os.getenv("GEMINI_API_KEY"),
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.client = genai.Client()

    def chat_completion(self, query: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name, contents=query
        )
        return (response.text or "").strip()


def llm_as_a_judge(client: Gemini_LLM_Client, judge_prompt: str, llm_inputs: str, response: str) -> tuple[int, str]:
    message = f"{judge_prompt}\n\nChat Context: {llm_inputs}\n\nResponse: {response}"
    raw = client.chat_completion(message)
    parsed = json.loads(raw)
    return parsed["rating"], parsed["rationale"]


# Module-level lazy singleton so the client is created at most once per process.
_client: Optional[Gemini_LLM_Client] = None


def _get_client() -> Gemini_LLM_Client:
    global _client
    if _client is None:
        _client = Gemini_LLM_Client()
    return _client


def compute_llm_metrics(predicted_response: str, context: str = "") -> dict:
    """Score a predicted response with the LLM judge.

    Args:
        predicted_response: The assistant response text to evaluate.
        context: Conversation history (user turns) shown to the judge as context.

    Returns:
        {"llm_score": int 1-5} if the response is non-empty, else {"llm_score": None}.
    """
    if not predicted_response:
        return {"llm_score": None}
    judge_prompt = open(_PROMPT_PATH).read()
    client = _get_client()
    try:
        score, _ = llm_as_a_judge(client, judge_prompt, context, predicted_response)
        return {"llm_score": int(score)}
    except Exception:
        return {"llm_score": None}
