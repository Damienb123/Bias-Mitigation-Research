import re
from collections import Counter

# normalize the text for readability and accuracy finding the top q keywords for each patient
def normalize_dx(text):
    if not text or not isinstance(text, str):
        return ""

    t = text.lower().strip()

    t = re.sub(r'\bunspecified\b', '', t)
    t = re.sub(r'\bwithout bleeding\b', '', t)
    t = re.sub(r'\binitial encounter\b', '', t)
    t = re.sub(r',.*$', '', t)
    t = re.sub(r'\s+', ' ', t).strip()

    if t == "":
        t = text.lower().split(",")[0].strip()

    return t


def extract_keywords(df, top_q=3):
    # DIAG_CODE stores the diagnosis for each patient
    diag_cols = [col for col in df.columns if "DIAG_CODE" in col]

    diagnoses = []
    for col in diag_cols:
        for dx in df[col].dropna():
            cleaned = normalize_dx(dx)
            if cleaned:
                diagnoses.append(cleaned)

    counts = Counter(diagnoses)
    return [item for item, _ in counts.most_common(top_q)]
