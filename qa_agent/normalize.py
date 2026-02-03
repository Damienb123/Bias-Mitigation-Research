import re

def normalize_dx(text):
    """
    Normalize diagnosis text to match knowledge-base entries.
    Example:
        'Dysphagia, unspecified' -> 'dysphagia'
        'Unspecified chronic gastritis without bleeding' -> 'chronic gastritis'
    """

    if not text:
        return text

    t = text.lower().strip()

    # Remove phrases like "unspecified", "without bleeding"
    t = re.sub(r'\bunspecified\b', '', t)
    t = re.sub(r'\bwithout bleeding\b', '', t)

    # Remove trailing descriptors like ", initial encounter"
    t = re.sub(r',.*$', '', t)

    # Remove extra whitespace
    t = re.sub(r'\s+', ' ', t).strip()

    return t
