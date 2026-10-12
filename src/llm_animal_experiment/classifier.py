import re
from typing import Literal

ResponseType = Literal["YES", "NO", "INVALID"]

def classify_response(response_text: str | None) -> ResponseType:
    """
    Classify whether a model response starts with 'Yes' or 'No' ('Nope' counts as No).
    
    If it starts with neither, returns 'INVALID'.
    Any leading non-word characters (whitespace, quotes, markdown, emoji) are stripped.
    """
    if not response_text:
        return "INVALID"
    
    cleaned = response_text.strip()
    # Strip leading markdown symbols, quotes, emoji, etc. (e.g. "**Yes", "'Yes'", "🎉 **Yes")
    cleaned = re.sub(r"^[\W_]+", "", cleaned)
    
    if re.match(r"^yes\b", cleaned, re.IGNORECASE):
        return "YES"
    elif re.match(r"^(no|nope)\b", cleaned, re.IGNORECASE):
        return "NO"
    return "INVALID"
