
import pandas as pd
import numpy as np
import category_encoders
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame
from datetime import datetime
from zipfile import ZipFile

desired_width=260
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',10)

'''
#with ZipFile("./df_2019-2020.zip",'r') as zipo:
    print(zipo.namelist())
    zipo.extractall()
'''


def basic_inf(data):
    print(data.head())
    print("DATA INFORMATION:")
    print("SOME STATS:")
    stats = data.describe(include='all')
    print(stats)
    print("DATA TYPES:")
    print(data.dtypes)
    print("NUMBER OF NULL VALUES:")
    print(data.isna().sum())
    print(data.shape)


df = pd.read_csv("./df_2019-2020.csv", delimiter=',')
print("BASIC INFO FOR the ORIGINAL DATA:")
basic_inf(df)
print("====" * 50)


'''
These are the columns I was playing with, feel free to add subtract columns of you wish
'''

my_col2 = ['CMPLNT_FR_DT',      'CMPLNT_FR_TM', 'CRM_ATPT_CPTD_CD', 'ADDR_PCT_CD', 'LAW_CAT_CD',
          'BORO_NM',     'LOC_OF_OCCUR_DESC', 'PREM_TYP_DESC', 'JURISDICTION_CODE',
           'VIC_AGE_GROUP', 'VIC_RACE', 'VIC_SEX']

my_col = ['CMPLNT_FR_DT',      'CMPLNT_FR_TM',   'ADDR_PCT_CD', 'KY_CD', 'OFNS_DESC',    'PD_CD',
          'CRM_ATPT_CPTD_CD',  'LAW_CAT_CD',     'BORO_NM',     'LOC_OF_OCCUR_DESC', 'PREM_TYP_DESC',
          'JURISDICTION_CODE', 'SUSP_AGE_GROUP', 'SUSP_RACE',   'SUSP_SEX',          'VIC_AGE_GROUP',
          'VIC_RACE',          'VIC_SEX']


df2 = df[my_col].dropna(axis=0, subset=my_col)
print("DF2 INFORMATION:")
print()
basic_inf(df2)


def column_desc(data):

    for cols in data.columns:
        print("COLUMN:", cols)
        print("UNIQUE VALUES:")
        print(data[cols].unique())
        print()
        print("VALUES COUNTS:")
        print(data[cols].value_counts())
        print()


column_desc(df2)
print("====" * 50)


def dropRows(data):
    drops = []
    for i, row in data.iterrows():
        if row["SUSP_AGE_GROUP"] not in ["<18", "18-24", "25-44", "45-64", "65+"]:
            drops.append(i)

        if row["SUSP_SEX"]       not in ["M","F"]:
            drops.append(i)

        if row["SUSP_RACE"]      not in ["BLACK","WHITE HISPANIC", "WHITE","BLACK HISPANIC",
                                         "ASIAN / PACIFIC ISLANDER, AMERICAN INDIAN/ALASKAN NATIVE"]:
            drops.append(i)

        if row["VIC_AGE_GROUP"]  not in ["<18", "18-24", "25-44", "45-64", "65+"]:
            drops.append(i)

        if row["VIC_SEX"]        not in ["M","F"]:
            drops.append(i)

        if row["VIC_RACE"]       not in ["BLACK","WHITE HISPANIC", "WHITE","BLACK HISPANIC",
                                         "ASIAN / PACIFIC ISLANDER, AMERICAN INDIAN/ALASKAN NATIVE"]:
            drops.append(i)

    return data.drop(drops).reset_index(drop=True)


df3 = dropRows(df2)


def dateConv(data):
    a = []
    b = []
    for i, row in data.iterrows():
        time_ob = datetime.strptime(row["CMPLNT_FR_TM"], "%H:%M:%S")
        date_ob = datetime.strptime(row["CMPLNT_FR_DT"], "%Y-%m-%d")
        a.append(time_ob.minute * 60 + time_ob.hour * 3600)
        b.append(date_ob.month * 100 + date_ob.day * 1)
    data['incdt_time'] = a
    data['incdt_date'] = b
    return data


def dateConv2(data):
    a = []
    b = []
    for i, row in data.iterrows():
        time_ob = datetime.strptime(row["CMPLNT_FR_TM"], "%H:%M:%S")
        date_ob = datetime.strptime(row["CMPLNT_FR_DT"], "%Y-%m-%d")
        if time_ob.hour in [20,21,22,23,0,1,2,3]:
            a.append(1)
        elif time_ob.hour in [4,5,6,7,8,9,10,11,12]:
            a.append(2)
        elif time_ob.hour in [13,14,15,16,17,18,19]:
            a.append(3)
        if date_ob.month in [5,6,7,8,]:
            b.append(1)
        elif date_ob.month in [1,2,3,4,9]:
            b.append(2)
    data['incdt_time'] = a
    data['incdt_date'] = b
    return data


df4 = dateConv2(df3)
print("DF3 INFORMATION:")
basic_inf(df4)
column_desc(df4)
print("====" * 50)


def grpRow(data):
    premis_group = []
    for i, row in data.iterrows():
        if row["PREM_TYP_DESC"] in ['RESIDENCE-HOUSE', 'MAILBOX OUTSIDE']:
            premis_group.append(1)

        elif row["PREM_TYP_DESC"] in ['RESIDENCE - PUBLIC HOUSING']:
            premis_group.append(2)

        elif row["PREM_TYP_DESC"] == ('STREET'):
            premis_group.append(3)

        elif row["PREM_TYP_DESC"] in ['MOSQUE','OTHER HOUSE OF WORSHIP','CHURCH', 'SYNAGOGUE']:
            premis_group.append(4)

        elif row["PREM_TYP_DESC"] in ['PARKING LOT/GARAGE (PUBLIC)', 'PARKING LOT/GARAGE (PRIVATE)', 'HIGHWAY/PARKWAY',
                                      'TUNNEL', 'BRIDGE','OPEN AREAS (OPEN LOTS)', 'CONSTRUCTION SITE',
                                      'STORAGE FACILITY', 'ABANDONED BUILDING', 'CEMETERY', 'HOMELESS SHELTER']:
            premis_group.append(5)

        elif row["PREM_TYP_DESC"] in ['BUS STOP','BUS (OTHER)', 'BUS (NYC TRANSIT)', 'FERRY/FERRY TERMINAL',
                                      'BUS TERMINAL','TRANSIT FACILITY (OTHER)','TAXI (LIVERY LICENSED)',
                                      'TAXI (YELLOW LICENSED)', 'AIRPORT TERMINAL','TRAMWAY',
                                      'TAXI/LIVERY (UNLICENSED)']:
            premis_group.append(6)

        elif row["PREM_TYP_DESC"] in ['HOSPITAL', 'DOCTOR/DENTIST OFFICE']:
            premis_group.append(7)

        elif row["PREM_TYP_DESC"] in ['ATM','BANK']:
            premis_group.append(8)

        elif row["PREM_TYP_DESC"] in ['RESTAURANT/DINER','BAR/NIGHT CLUB', 'HOTEL/MOTEL','MARINA/PIER',
                                      'SOCIAL CLUB/POLICY']:
            premis_group.append(9)

        elif row["PREM_TYP_DESC"] in ['PUBLIC BUILDING', 'GAS STATION']:
            premis_group.append(10)

        elif row["PREM_TYP_DESC"] in ['PRIVATE/PAROCHIAL SCHOOL', 'PUBLIC SCHOOL','DAYCARE FACILITY',
                                      'PARK/PLAYGROUND']:
            premis_group.append(11)

        elif row["PREM_TYP_DESC"] in ['RESIDENCE - APT. HOUSE']:
            premis_group.append(12)

        else:
            premis_group.append(30)

    data['premis_var'] = premis_group
    return data




df5 = grpRow(df4)
print("DF5 INFORMATION:")
print(df5.head())
print("====" * 50)


def encoder(data):

###### BINARY  ENCODER ######'
    B_encoder = category_encoders.BinaryEncoder(cols=['premis_var'])
    data = B_encoder.fit_transform(data)
    B_encoder = category_encoders.BinaryEncoder(cols=['incdt_time'])
    data = B_encoder.fit_transform(data)
    B_encoder = category_encoders.BinaryEncoder(cols=['incdt_date'])
    data = B_encoder.fit_transform(data)

###### DATA STANDARDISATION ######
#    sc = StandardScaler()
#    data['incdt_time'] = sc.fit_transform(data['incdt_time'].values.reshape(-1,1))
#    data['incdt_date'] = sc.fit_transform(data['incdt_date'].values.reshape(-1,1))

###### ONE HOT ENCODER ######
    OH_encoder = OneHotEncoder()
    hc1 = DataFrame(OH_encoder.fit_transform(data['BORO_NM'].values.reshape(-1,1)).toarray(),
                    columns = ['BORO1', 'BORO2','BORO3', 'BORO4', 'BORO5'])
    hc2 = DataFrame(OH_encoder.fit_transform(data['VIC_AGE_GROUP'].values.reshape(-1, 1)).toarray(),
                    columns=['VIC_AGE1', 'VIC_AGE2', 'VIC_AGE3', 'VIC_AGE4', 'VIC_AGE5'])
    hc3 = DataFrame(OH_encoder.fit_transform(data['LOC_OF_OCCUR_DESC'].values.reshape(-1, 1)).toarray(),
                    columns=['LOC_DESC1', 'LOC_DESC2', 'LOC_DESC3', 'LOC_DESC4'])
    hc4 = DataFrame(OH_encoder.fit_transform(data['VIC_RACE'].values.reshape(-1, 1)).toarray(),
                    columns=['VIC_RACE1', 'VIC_RACE2', 'VIC_RACE3', 'VIC_RACE4'])
    hc5 = DataFrame(OH_encoder.fit_transform(data['VIC_SEX'].values.reshape(-1, 1)).toarray(),
                    columns=['VICM_SEX1', 'VICM_SEX2'])
    hc6 = DataFrame(OH_encoder.fit_transform(data['SUSP_RACE'].values.reshape(-1, 1)).toarray(),
                    columns=['SUSP_RACE1', 'SUSP_RACE2', 'SUSP_RACE3', 'SUSP_RACE4'])
    hc7 = DataFrame(OH_encoder.fit_transform(data['VIC_SEX'].values.reshape(-1, 1)).toarray(),
                    columns=['SUSP_SEX1', 'SUSP_SEX2'])

    data = pd.concat([data,hc1,hc2,hc3,hc4,hc5,hc6,hc7], axis=1)
    return data


df6 = encoder(df5)
print("DF6 INFORMATION:")
print(df6.head())
print(basic_inf(df6))
print(column_desc(df6))
print("====" * 50)


##################################################################################
# After this point, it is about picking and choosing which variables to use in
# modeling so will be deleting already encoded variables and encoding
# my dependent variable.
##################################################################################

col_del = ['CMPLNT_FR_DT',      'CMPLNT_FR_TM',       'ADDR_PCT_CD',
          'CRM_ATPT_CPTD_CD'  ,  'BORO_NM',           'LOC_OF_OCCUR_DESC', 'PREM_TYP_DESC',
          'JURISDICTION_CODE',   'SUSP_AGE_GROUP',    'SUSP_RACE',        'SUSP_SEX',          'VIC_AGE_GROUP',
          'VIC_RACE',            'VIC_SEX',           'SUSP_RACE1', 'SUSP_RACE2', 'SUSP_RACE3', 'SUSP_RACE4',
          'SUSP_SEX1', 'SUSP_SEX2', 'incdt_time_0', 'premis_var_0']

df7 = df6.drop(columns=col_del)

grps = []
for i,row in df7.iterrows():
    if row['LAW_CAT_CD'] == 'FELONY':
        grps.append(0)
    elif row['LAW_CAT_CD'] == 'MISDEMEANOR':
        grps.append(1)
    elif row['LAW_CAT_CD'] == 'VIOLATION':
        grps.append(2)

df7['OFFNS'] = grps
df8 = df7.drop(columns='LAW_CAT_CD')

df8['PDCD_CODE'] = (df8['PD_CD']/100).astype('int') * 10
df8['KYCD_CODE'] = (df8['KY_CD']/10).astype('int')
df8.to_csv('cleaned.csv')