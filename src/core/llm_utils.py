import json
import re
from typing import Any, Dict, List, Optional

# Regular expressions used when sanitizing LLM output
_CODE_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")


def _sanitize_control_chars(payload: str) -> str:
    """Replace disallowed control characters with unicode escapes."""

    def _replacement(match: re.Match[str]) -> str:
        return f"\\u{ord(match.group(0)):04x}"

    return _CONTROL_CHAR_PATTERN.sub(_replacement, payload)


def _extract_json_candidates(raw_text: str) -> List[str]:
    """Return possible JSON payloads contained within the LLM response."""

    candidates: List[str] = []
    if not raw_text:
        return candidates

    trimmed = raw_text.strip()
    if trimmed:
        candidates.append(trimmed)

    for block in _CODE_FENCE_PATTERN.findall(trimmed):
        block_trimmed = block.strip()
        if block_trimmed:
            candidates.append(block_trimmed)

    # Balanced brace extraction to capture embedded JSON objects
    depth = 0
    start_index: Optional[int] = None
    for index, char in enumerate(trimmed):
        if char == "{":
            if depth == 0:
                start_index = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start_index is not None:
                    block = trimmed[start_index : index + 1].strip()
                    if block:
                        candidates.append(block)
                    start_index = None

    # Deduplicate while preserving order
    seen = set()
    unique_candidates: List[str] = []
    for candidate in candidates:
        if candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)

    return unique_candidates


def parse_llm_json_response(raw_text: str) -> Dict[str, Any]:
    """Best-effort parsing of a JSON payload returned by an LLM."""

    if not raw_text or not raw_text.strip():
        raise ValueError("Unable to extract JSON object from empty LLM response")

    for candidate in _extract_json_candidates(raw_text):
        sanitized = _sanitize_control_chars(candidate)
        try:
            return json.loads(sanitized, strict=False)
        except json.JSONDecodeError:
            try:
                return json.loads(_sanitize_control_chars(candidate.strip()), strict=False)
            except json.JSONDecodeError:
                pass
            # Try trimming trailing text after the last closing brace
            end_index = sanitized.rfind("}")
            if end_index != -1 and end_index + 1 < len(sanitized):
                try:
                    return json.loads(sanitized[: end_index + 1], strict=False)
                except json.JSONDecodeError:
                    pass
            continue

    raise ValueError("Unable to extract JSON object from LLM response")


def extract_structured_decision(response: Dict[str, Any]) -> Dict[str, str]:
    decision = response.get("decision", "DECLINE").upper()
    reasoning = response.get("reasoning", "No reasoning provided.")
    thought_process = response.get("thought_process", "No thought process provided.")
    return {
        "decision": decision,
        "reasoning": reasoning,
        "thought_process": thought_process
    }


def extract_structured_review(response: Dict[str, Any]) -> Dict[str, Any]:
    review_content = response.get("review_content")
    thought_process = response.get("thought_process", "No thought process provided.")
    raw_response = response.get("raw_response")

    if isinstance(review_content, str):
        try:
            review_content = json.loads(review_content)
        except json.JSONDecodeError:
            review_content = {"unparsed": review_content}

    if review_content is None:
        review_content = {}

    return {
        "review_content": review_content,
        "thought_process": thought_process,
        "raw_response": raw_response,
    }
