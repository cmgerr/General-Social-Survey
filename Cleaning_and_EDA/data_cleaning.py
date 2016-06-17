import pandas as pd
import numpy as np

# set filepath to csv
filepath = '../../gss.csv'

# function to print value counts for all columns
def printvalcounts(df):
    for col in df:
        print col
        print df[col].value_counts()

# first, read in the descriptions and variable names for all columns
descriptions = pd.read_csv(filepath, nrows=1)
print descriptions
# create a dictionary of variable definitions to use as reference if needed
desc_dict = {}
for col in descriptions:
    desc_dict[descriptions[col][0]] = col

# get full list of columns for reference as needed
test = pd.read_csv(filepath, header=1, nrows=5)
cols = test.columns.tolist()
valid_cols = [x for x in cols if "Unnamed" not in x]
valid_cols

# choose a subset of columns to extract for analysis
col_subset = ['year', 'marital', 'sibs', 'childs',
    'age', 'educ', 'paeduc', 'maeduc', 'speduc', 'sex', 'hompop',
    'income', 'earnrs', 'happy', 'polviews', 'babies', 'preteen',
    'teens', 'adults', 'divorce', 'health', 'famgen', 'dwelown',
    'goodlife', 'weekswrk', 'satfin','satjob', 'dwelling', 'hhrace']

df = pd.read_csv(filepath, header=1, usecols = col_subset)

# look at info
df.info()

# look at values for each column (validate against GSS website summary)
printvalcounts(df[df['year']==2014])


# very oddly, it seems that all the zero codes are recorded as NaN values. so,
# replace all NaN values with zero first, then go back through and replace
# ACTUAL codes that indicate no response or n/a response as np.nan

df.fillna(0,inplace=True)

# convert marital status and sex to categories, replace
# "didn't answer"/"don't know" type codes for other numeric variables
marital_status = {
    1: 'married',
    2: 'widowed',
    3: 'divorced',
    4: 'separated',
    5: 'never_married',
    9: np.nan
}

replace_dict = {
    'marital': marital_status,
    'sibs': {-1: np.nan, 98: np.nan, 99: np.nan},
    'childs': {9: np.nan},
    'educ': {98: np.nan, 99: np.nan},
    'paeduc': {97: np.nan, 98: np.nan, 99: np.nan},
    'maeduc': {97: np.nan, 98: np.nan, 99: np.nan},
    'speduc': {97: np.nan, 98: np.nan, 99: np.nan},
    'sex': {1: 'Male', 2: 'Female'},
    'hompop': {99: np.nan},
    'income': {13: np.nan, 0: np.nan, 98: np.nan, 99: np.nan},
    'earnrs': {9: np.nan},
    'happy': {0: np.nan, 8: np.nan, 9: np.nan},
    'polviews':{8:4, 9: np.nan, 0:np.nan},
    # bizarrely there are huge numbers of N/A responses in 1983, 2002, 2004
    # need to bear this in mind in our modeling...
    # Also, I'm making an assumption that "don't know" (code 8) can be
    # interpreted as equal to "moderate" (code 4).
    'babies':{9: np.nan},
    'preteen': {9:np.nan},
    'teens':{9: np.nan},
    'adults':{9: np.nan},
    'divorce':{1: 'yes', 2:'no', 8: np.nan, 9: np.nan, 0: 'no'},
    # I am making a (rather large) assumption here that the large number of N/A
    # responses are from never-married respondents...
    'health': {8: np.nan, 9: np.nan, 0: np.nan}, # no data for 78,83,86
    'famgen':{0:np.nan},
    'dwelown':{0:3, 8: np.nan, 9: np.nan},
    # only 85-present
    # replaced 0(not applicable) with (other) as I doubt they surveyed 1.8k homeless people...)
    'goodlife':{8: np.nan, 9: np.nan, 0: np.nan},
    # very strange amount of n/a for this question... may be worth skipping?
    'weekswrk':{-1:np.nan, 99:np.nan, 98:np.nan},
    #31k of these values were -1, which I can assume was 0 weeks worked?
    # 94-present
    'satfin':{8: np.nan, 9: np.nan, 0: np.nan},
    'satjob':{8: np.nan, 9: np.nan, 0: np.nan},
    'dwelling':{98: np.nan, 99: np.nan, 0: np.nan},
    'hhrace':{8: np.nan, 9: np.nan, 0: np.nan}
}

df.replace(to_replace = replace_dict, inplace=True)

# look at info again
df.info()

# check value counts (validate against GSS website summary)
printvalcounts(df[df['year']==2014])



# export to csv
df.to_csv('../Data/gss_subset_cleaned.csv', encoding='utf-8')
