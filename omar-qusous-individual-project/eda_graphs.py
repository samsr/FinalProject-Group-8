import pandas as pd
import datetime
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

plt.rc('font', **font)

# for NYPD_Complaint_Data_Historic.csv go to:
# https://drive.google.com/drive/folders/1KiBFiLR820aRWtO8gKQVUH6hJGqYf65d?usp=sharing
# file size is too big to be uploaded to github

df_ny = pd.read_csv('NYPD_Complaint_Data_Historic.csv')

df_ny.loc[0:10,'CMPLNT_FR_DT']

df_ny = df_ny[df_ny['CMPLNT_FR_DT'].apply(lambda x: isinstance(x, str))]

df_ny['CMPLNT_FR_DT'] = pd.to_datetime(df_ny['CMPLNT_FR_DT'], errors = 'coerce')

df_ny = df_ny[(df_ny['CMPLNT_FR_DT'] > datetime.date(2010,1,1))]

df_ny.head()

df_ny.columns

df_ny.dropna(axis = 0, subset=['CMPLNT_NUM', 'CMPLNT_FR_DT','KY_CD', 'OFNS_DESC','LAW_CAT_CD', 'BORO_NM',
                               'PREM_TYP_DESC','VIC_AGE_GROUP', 'VIC_RACE', 'VIC_SEX'], inplace=True)

df_ny = df_ny[df_ny.groupby('VIC_SEX')['VIC_SEX'].transform('size') > 260000]

df_ny = df_ny[df_ny.groupby('VIC_AGE_GROUP')['VIC_AGE_GROUP'].transform('size') > 52000]

df_ny_grp1 = df_ny.groupby([df_ny['CMPLNT_FR_DT'].dt.to_period('M'), df_ny['VIC_SEX']] ).count()

df_ny.columns

df_ny = df_ny[['CMPLNT_NUM', 'CMPLNT_FR_DT', 'OFNS_DESC', 'PD_CD', 'LAW_CAT_CD', 'BORO_NM',
       'PREM_TYP_DESC', 'VIC_AGE_GROUP', 'VIC_RACE', 'VIC_SEX']]

# df_ny.to_csv('df_ny.csv')
# pickle.dump( df_ny, open( "df_ny.p", "wb" ) )

# Plot 1
fig, ax = plt.subplots(figsize=(15,7))
df_ny.groupby([df_ny['CMPLNT_FR_DT'].dt.to_period('M'), df_ny['VIC_SEX']] ).count()['CMPLNT_NUM'].unstack().plot(ax=ax)
plt.title("Number of crimes commited onto Females and Males from 2010 to 2019")

# PLot 2
fig, ax = plt.subplots(figsize=(15,7))
df_ny.groupby([df_ny['CMPLNT_FR_DT'].dt.to_period('M'), df_ny['VIC_AGE_GROUP']] ).count()['CMPLNT_NUM'].unstack().plot(ax=ax)
plt.title("Number of crimes commited onto various age groups from 2010 to 2019")

# Plot 3
fig, ax = plt.subplots(figsize=(15,7))
df_ny.groupby([df_ny['CMPLNT_FR_DT'].dt.to_period('M'), df_ny['VIC_AGE_GROUP']] ).count()['CMPLNT_NUM'].unstack().plot(ax=ax)
plt.title("Number of crimes commited onto various age groups from 2010 to 2019")

# Plot 4
fig, ax = plt.subplots(figsize=(15,7))
df_ny.groupby([df_ny['CMPLNT_FR_DT'].dt.to_period('M'), df_ny['BORO_NM']] ).count()['CMPLNT_NUM'].unstack().plot(ax=ax)
plt.title("Number of crimes commited in the five NY boroughs from 2010 to 2019")

# Plot 5
crimes_group_df_ny=df_ny.copy()
crimes_group_df_ny = (crimes_group_df_ny.groupby([df_ny['CMPLNT_FR_DT'].dt.to_period('M').rename('dates'), crimes_group_df_ny['OFNS_DESC'].rename('crimesz')]).count()).sort_values(['dates','CMPLNT_NUM'], ascending=False)[0:11]
crimes_group_df_ny = pd.DataFrame(crimes_group_df_ny)
crimes_group_df_ny = crimes_group_df_ny.reset_index()
crimes_group_df_ny.describe
(df_ny.groupby('OFNS_DESC').count()).sort_values('CMPLNT_NUM', ascending=False)
crimes_group_df_ny = df_ny[df_ny.groupby('OFNS_DESC')['OFNS_DESC'].transform('size') > 100000]
fig, ax = plt.subplots(figsize=(15,7))
crimes_group_df_ny.groupby([df_ny['CMPLNT_FR_DT'].dt.to_period('M'), df_ny['OFNS_DESC']] ).count()['CMPLNT_NUM'].unstack().plot(ax=ax)
plt.title("Ten most recurring crimesfrom 2010 to 2019")
plt.legend(fontsize=10)

# Set 2

df = pd.read_csv('data_cleaned_for_model.csv')
plt.figure(figsize=(10, 6))
ax = plt.axes()
sns.heatmap(df.isna(), cbar=False)
ax.set_title('Nan Heatmap Plot')
plt.show()

pd.DataFrame(df.info())

month_group_df = df.groupby(df['CMPLNT_FR_DT'].apply(lambda x : x[0])).count()
plt.figure(figsize=(10, 6))
plt.bar(month_group_df.index, month_group_df['CMPLNT_NUM'], facecolor='blue')
plt.xlabel('Month')
plt.ylabel('Count')
plt.title(r'Number of crimes occuring each month in the year 2019')
plt.show()

crime_group_df = df.groupby(df['LAW_CAT_CD'].apply(lambda x : x)).count()
plt.figure(figsize=(10, 6))
plt.bar(crime_group_df.index, crime_group_df['LAW_CAT_CD'], facecolor='blue')
plt.xlabel('Crime class')
plt.ylabel('Count')
plt.title(r'Number of Felonies, Violations and Misdemnors (Year 2019)')
plt.show()

vicAge_group_df=df.copy()
vicAge_group_df = (vicAge_group_df.groupby(vicAge_group_df['VIC_AGE_GROUP']).count()).sort_values('CMPLNT_NUM', ascending=False)[0:6]
plt.figure(figsize=(10, 6))
plt.bar(vicAge_group_df.index, vicAge_group_df['CMPLNT_NUM'], facecolor='blue')
plt.xlabel("Victim's age group")
plt.ylabel('Count')
plt.title(r'Number of crimes committed to various victim age groups (Year 2019)')
plt.show()

vicSex_group_df=df.copy()
vicSex_group_df = (vicSex_group_df.groupby(vicSex_group_df['VIC_SEX']).count()).sort_values('CMPLNT_NUM', ascending=False)[0:2]
plt.figure(figsize=(10, 6))
plt.bar(vicSex_group_df.index, vicSex_group_df['CMPLNT_NUM'], facecolor='blue')
plt.xlabel("Victim's Sex" )
plt.ylabel('Count')
plt.title(r'Number of crimes committed to females and males (Year 2019)')
plt.show()

Loc_group_df=df.copy()
Loc_group_df = (Loc_group_df.groupby(Loc_group_df['PREM_TYP_DESC']).count()).sort_values('CMPLNT_NUM', ascending=False)[0:11]
plt.figure(figsize=(10, 6))
plt.bar(Loc_group_df.index, Loc_group_df['CMPLNT_NUM'], facecolor='blue')
plt.xticks(rotation=90)
plt.xlabel("Crime type")
plt.ylabel('Count')
plt.title(r'Ten most occuring crimes in the year 2019')
plt.show()

crimes_group_df=df.copy()
crimes_group_df = (crimes_group_df.groupby(crimes_group_df['OFNS_DESC']).count()).sort_values('CMPLNT_NUM', ascending=False)[0:11]
plt.figure(figsize=(10, 6))
plt.bar(crimes_group_df.index, crimes_group_df['CMPLNT_NUM'], facecolor='blue')
plt.xticks(rotation=90)
plt.xlabel('Premesis')
plt.ylabel('Count')
plt.title(r'Ten most places were crimes are comitted for the year 2019')
plt.show()

boro_group_df=df.copy()
boro_group_df = (boro_group_df.groupby(boro_group_df['BORO_NM']).count()).sort_values('CMPLNT_NUM', ascending=False)[0:]
plt.figure(figsize=(10, 6))
plt.bar(boro_group_df.index, boro_group_df['CMPLNT_NUM'], facecolor='blue')
plt.xticks(rotation=90)
plt.xlabel('New York Borough')
plt.ylabel('Count')
plt.title(r'Number of crimes commited in the five New York boroughs')
plt.show()

