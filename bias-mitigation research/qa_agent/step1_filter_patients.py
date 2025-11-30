# focuses on filtering patient data by age_group, gender, race, zip

# import for pandas
import pandas as pd


# read csv data for patient info
def pd.read_csv("data/patients.csv")

# only apply filters that exist in the input
# input patient data for given patient in their grouping
for key in ["age_group", "gender", "race", "zip"]:
    if key in patient_info:
        df = df[df[key] == patient.info[key]]

# return file
return df