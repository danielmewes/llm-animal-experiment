import re
from typing import Literal

ResponseType = Literal["YES", "NO", "INVALID"]

def classify_response(response_text: str | None) -> ResponseType:
    """
    Classify whether a model response starts with 'Yes' or 'No'.
    
    If it starts with neither, returns 'INVALID'.
    Leading whitespace, quotes, and markdown formatting are stripped.
    """
    if not response_text:
        return "INVALID"
    
    cleaned = response_text.strip()
    # Strip leading markdown symbols, quotes, or asterisks (e.g. "**Yes", "*Yes*", "'Yes'")
    cleaned = re.sub(r"^[\s\"'*_#`]+", "", cleaned)
    
    if re.match(r"^yes\b", cleaned, re.IGNORECASE):
        return "YES"
    elif re.match(r"^no\b", cleaned, re.IGNORECASE):
        return "NO"
    return "INVALID"
