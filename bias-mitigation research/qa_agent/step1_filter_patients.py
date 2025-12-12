import pandas as pd
# extract patient info iteratively through each column of data
def filter_patients(df, patient_info):
    cohort = df.copy()
    # patient demographics captured are assigned for each of the 4 info pieces
    for key in ["SEX_CODE", "PAT_AGE", "RACE", "PAT_ZIP"]:
        if key in patient_info and patient_info[key]:
            cohort = cohort[cohort[key] == patient_info[key]]

    return cohort
