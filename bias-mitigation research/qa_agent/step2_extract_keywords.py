# Extracts keywords like diseases, procedures, prescriptions for this given case

# imports
from collections import Counter

# extract_keywords function containing diseases, procedures, prescriptions
def extract_keywords(cohort_df, top_q=3):
    items = list(cohort_df["disease"].dropna().values) \
          + list(cohort_df["procedures"].dropna().values) \
          + list(cohort_df["prescriptions"].dropna().values)

    # counts keywords within most common q
    counts = Counter(items)
    keywords = [item for item, _ in counts.most_common(top_q)]
    return keywords