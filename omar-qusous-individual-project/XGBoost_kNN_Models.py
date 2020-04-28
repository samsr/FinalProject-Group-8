# Imports
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import time
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.decomposition import TruncatedSVD
import itertools
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import xgboost as xgb

df = pd.read_csv('data_cleaned_for_model.csv')
print(df.head())
print(df.columns)
sns.heatmap(df.isna(), cbar=False)

df.drop(['Unnamed: 0', 'Unnamed: 0.1', 'CMPLNT_NUM', 'CMPLNT_FR_DT_CODED', 'CMPLNT_TO_TM', 'CMPLNT_TO_DT',
        'isWHITE HISPANIC', 'isWHITE', 'isBLACK', 'isUNKNOWN', 'isBLACK HISPANIC', 'isASIAN / PACIFIC ISLANDER',
        'isAMERICAN INDIAN/ALASKAN NATIVE', 'VIC_RACE','Latitude', 'Longitude', 'SUSP_AGE_GROUP_CODED', 'STATION_NAME',
        'CRM_ATPT_CPTD_CD', 'X_COORD_CD', 'Y_COORD_CD', 'RPT_DT', "PD_CD", 'PD_DESC', 'PATROL_BORO',
        'SUSP_RACE', 'PARKS_NM', 'HADEVELOPT', 'JURISDICTION_CODE', 'Lat_Lon','JURIS_DESC', 'HOUSING_PSA', 'LOC_OF_OCCUR_DESC',
        'SUSP_AGE_GROUP', 'SUSP_SEX', 'TRANSIT_DISTRICT', 'ADDR_PCT_CD', 'isnan'], axis=1, inplace=True)

df.dropna(inplace=True)

# Categorizing 'PREM_TYP_DESC'
df['PREM_TYP_DESC'].value_counts()
df.drop(df[df['PREM_TYP_DESC'] == 'OTHER'].index, axis=0, inplace=True)
df.loc[(df['PREM_TYP_DESC'] == 'RESIDENCE-HOUSE') | (df['PREM_TYP_DESC'] == 'MAILBOX OUTSIDE') |
       (df['PREM_TYP_DESC'] == 'RESIDENCE - PUBLIC HOUSING') | 
       (df['PREM_TYP_DESC'] == 'PARKING LOT/GARAGE (PRIVATE)') | 
       (df['PREM_TYP_DESC'] == 'RESIDENCE - APT. HOUSE'), 
       'PREM_TYP'] = 'RESIDENCE'

df.loc[(df['PREM_TYP_DESC'] == 'MOSQUE') | (df['PREM_TYP_DESC'] == 'OTHER HOUSE OF WORSHIP') |
       (df['PREM_TYP_DESC'] == 'CHURCH') | (df['PREM_TYP_DESC'] == 'SYNAGOGUE'), 
       'PREM_TYP'] = 'HOUSE OF WORSHIP'

df.loc[(df['PREM_TYP_DESC'] == 'STREET') | (df['PREM_TYP_DESC'] == 'HIGHWAY/PARKWAY') |
       (df['PREM_TYP_DESC'] == 'BRIDGE') | (df['PREM_TYP_DESC'] == 'TUNNEL'), 
       'PREM_TYP'] = 'ST.Rd.Hw.Brdg'
        
df.loc[(df['PREM_TYP_DESC'] == 'PARKING LOT/GARAGE (PUBLIC)') | (df['PREM_TYP_DESC'] == 'OPEN AREAS (OPEN LOTS)'),
       'PREM_TYP'] = 'PUBLIC PARKING LOTS'

df.loc[(df['PREM_TYP_DESC'] == 'CONSTRUCTION SITE'), 
       'PREM_TYP'] = 'CONSTRUCTION SITE'

df.loc[(df['PREM_TYP_DESC'] == 'ABANDONED BUILDING'), 
       'PREM_TYP'] = 'ABANDONED BUILDING' 

df.loc[(df['PREM_TYP_DESC'] == 'CEMETERY'), 
       'PREM_TYP'] = 'CEMETERY'
        
df.loc[(df['PREM_TYP_DESC'] == 'HOMELESS SHELTER'), 
       'PREM_TYP'] = 'HOMELESS SHELTER'
df.loc[(df['PREM_TYP_DESC'] == 'BUS STOP') | (df['PREM_TYP_DESC'] == 'BUS (OTHER)') |
       (df['PREM_TYP_DESC'] == 'BUS (NYC TRANSIT)') | (df['PREM_TYP_DESC'] == 'TRANSIT FACILITY (OTHER)') |
       (df['PREM_TYP_DESC'] == 'FERRY/FERRY TERMINAL') | (df['PREM_TYP_DESC'] == 'BUS TERMINAL') |
       (df['PREM_TYP_DESC'] == 'AIRPORT TERMINAL') | (df['PREM_TYP_DESC'] == 'TAXI/LIVERY (UNLICENSED)') |
       (df['PREM_TYP_DESC'] == 'TRAMWAY') | (df['PREM_TYP_DESC'] == 'TRANSIT - NYC SUBWAY'), 
       'PREM_TYP'] = 'PUBLIC.PRIVATE TRANSPORT'
df.loc[(df['PREM_TYP_DESC'] == 'HOSPITAL') | 
       (df['PREM_TYP_DESC'] == 'DOCTOR/DENTIST OFFICE'), 
       'PREM_TYP'] = 'MEDICAL FACILITY'

df.loc[(df['PREM_TYP_DESC'] == 'ATM') | (df['PREM_TYP_DESC']== 'CHECK CASHING BUSINESS') 
       | (df['PREM_TYP_DESC'] == 'LOAN COMPANY'), 'PREM_TYP'] = 'BANK'

df.loc[(df['PREM_TYP_DESC'] == 'RESTAURANT/DINER') | (df['PREM_TYP_DESC'] == 'BAR/NIGHT CLUB') | 
       (df['PREM_TYP_DESC'] == 'HOTEL/MOTEL') | (df['PREM_TYP_DESC'] == 'MARINA/PIER') |
       (df['PREM_TYP_DESC'] == 'SOCIAL CLUB/POLICY') | (df['PREM_TYP_DESC'] == 'FAST FOOD'), 
       'PREM_TYP'] = 'RESTAURANT.BAR.HOTEL'
        
df.loc[(df['PREM_TYP_DESC'] == 'DEPARTMENT STORE') | (df['PREM_TYP_DESC'] == 'CHAIN STORE') |
       (df['PREM_TYP_DESC'] == 'CLOTHING/BOUTIQUE') | (df['PREM_TYP_DESC'] == 'TELECOMM. STORE') |
       (df['PREM_TYP_DESC'] == 'VARIETY STORE') | (df['PREM_TYP_DESC'] == 'JEWELRY') |
       (df['PREM_TYP_DESC'] == 'SMALL MERCHANT'),
       'PREM_TYP'] = 'SHOPPING'
       
df.loc[(df['PREM_TYP_DESC'] == 'GROCERY/BODEGA') | (df['PREM_TYP_DESC'] == 'LIQUOR STORE') |
        (df['PREM_TYP_DESC'] == 'STORE UNCLASSIFIED') | (df['PREM_TYP_DESC'] == 'DRY CLEANER/LAUNDRY') |
        (df['PREM_TYP_DESC'] == 'Gas_Station') | (df['PREM_TYP_DESC'] == 'DRUG STORE') |
        (df['PREM_TYP_DESC'] == 'CANDY STORE'), 
        'PREM_TYP'] = 'GROCERY SHOPPING'
        
df.loc[(df['PREM_TYP_DESC'] == 'BEAUTY & NAIL SALON') | (df['PREM_TYP_DESC'] == 'GYM/FITNESS FACILITY'),
       'PREM_TYP'] = 'SPA.BEAUTYSALON.GYM'        

df.loc[(df['PREM_TYP_DESC'] == 'COMMERCIAL BUILDING') | (df['PREM_TYP_DESC'] == 'STORAGE FACILITY') |
       (df['PREM_TYP_DESC'] == 'FACTORY/WAREHOUSE'),
       'PREM_TYP'] = 'COMMERCIAL BUILDING'     

df.loc[(df['PREM_TYP_DESC'] == 'PRIVATE/PAROCHIAL SCHOOL') | (df['PREM_TYP_DESC'] == 'PUBLIC SCHOOL') |
       (df['PREM_TYP_DESC'] == 'DAYCARE FACILITY') | (df['PREM_TYP_DESC'] == 'PARK/PLAYGROUND'),
       'PREM_TYP'] = 'SCHOOL.PLAYGROUND'

df['PREM_TYP'].value_counts()

df.drop(['PREM_TYP_DESC'], axis=1, inplace=True)

# Categorizing 'OFNS_DESC'

df_gp = df.groupby(['KY_CD','OFNS_DESC']).count().sort_values(by=['CMPLNT_FR_DT'], ascending=False)
df_gp.reset_index(level=None, drop=False, inplace=True, col_level=0, col_fill='')
df_gp

df_gp.loc[df_gp.OFNS_DESC.str.contains('MISCELLANEOUS PENAL LAW'), :]

df = df[df.groupby('CMPLNT_FR_DT')['CMPLNT_FR_DT'].transform('size') > 100]

list1 = df_gp[df_gp['CMPLNT_FR_DT']>100]['OFNS_DESC'].values

list2 = ['PETIT LARCENY', 'GRAND LARCENY', 'ROBBERY', 'BURGLARY', "THEFT OF SERVICES", "BURGLAR'S TOOLS", 
        'POSSESSION OF STOLEN PROPERTY', 'OTHER OFFENSES RELATED TO THEF', 'GRAND LARCENY OF MOTOR VEHICLE',
        'ASSAULT 3 & RELATED OFFENSES', 'FELONY ASSAULT', 'OFFENSES AGAINST THE PERSON', 'HARRASSMENT 2',
        'CRIMINAL MISCHIEF & RELATED OF', 'OFF. AGNST PUB ORD SENSBLTY &', 'DANGEROUS DRUGS', 
        'OFFENSES AGAINST PUBLIC ADMINI', 'INTOXICATED & IMPAIRED DRIVING', 'VEHICLE AND TRAFFIC LAWS',
        'MISCELLANEOUS PENAL LAW', 'RAPE', 'SEX CRIMES', 'OFFENSES INVOLVING FRAUD', 'FRAUDS', 'THEFT-FRAUD', 
        'FRAUDULENT ACCOSTING', 'NYS LAWS-UNCLASSIFIED FELONY', 'DANGEROUS WEAPONS', 'CRIMINAL TRESPASS',
        'FORGERY', 'UNAUTHORIZED USE OF A VEHICLE', 'ADMINISTRATIVE CODE', 'ARSON', 'GAMBLING', 'OTHER STATE LAWS (NON PENAL LA']

[x for x in list1 if x not in list2]

df.loc[(df['OFNS_DESC'] == 'PETIT LARCENY') | (df['OFNS_DESC'] == 'GRAND LARCENY') |
       (df['OFNS_DESC'] == 'ROBBERY') | (df['OFNS_DESC'] == 'BURGLARY') | 
       (df['OFNS_DESC'] == "BURGLAR'S TOOLS") | (df['OFNS_DESC'] == 'THEFT OF SERVICES') |
       (df['OFNS_DESC'] == 'POSSESSION OF STOLEN PROPERTY') | (df['OFNS_DESC'] == 'OTHER OFFENSES RELATED TO THEF') |
       (df['OFNS_DESC'] == 'GRAND LARCENY OF MOTOR VEHICLE'), 'CRIME'] = 'THEFT/LARCENY'

df.loc[(df['OFNS_DESC'] == 'ASSAULT 3 & RELATED OFFENSES') | (df['OFNS_DESC'] == 'FELONY ASSAULT') |
       (df['OFNS_DESC'] == 'OFFENSES AGAINST THE PERSON'), 'CRIME'] = 'ASSAULT'

df.loc[(df['OFNS_DESC'] == 'HARRASSMENT 2'), 'CRIME'] = 'HARRASSMENT'

df.loc[(df['OFNS_DESC'] == 'CRIMINAL MISCHIEF & RELATED OF'), 'CRIME'] = 'CRIMINAL MISCHIEF'

df.loc[(df['OFNS_DESC'] == 'OFF. AGNST PUB ORD SENSBLTY &')| (df['OFNS_DESC'] == 'OFFENSES AGAINST PUBLIC ADMINI'),
       'CRIME'] = 'PUBLIC DISORDER'

df.loc[(df['CMPLNT_FR_DT'] == 'DANGEROUS DRUGS'), 'CRIME'] = 'DANGEROUS DRUGS'

df.loc[(df['OFNS_DESC'] == 'THEFT-FRAUD') | (df['OFNS_DESC'] == 'FRAUDS') |
       (df['OFNS_DESC'] == 'OFFENSES INVOLVING FRAUD') | (df['OFNS_DESC'] == 'FRAUDULENT ACCOSTING') ,
       'CRIME'] = 'FRAUD'

df.loc[(df['OFNS_DESC'] == 'SEX CRIMES') | (df['OFNS_DESC'] == 'RAPE'),
       'CRIME'] = 'SEX CRIMES'

df.loc[(df['OFNS_DESC'] == 'DANGEROUS WEAPONS'),
       'CRIME'] = 'DANGEROUS WEAPONS'
       
df.loc[(df['OFNS_DESC'] == 'CRIMINAL TRESPASS'),
       'CRIME'] = 'CRIMINAL TRESPASS'
       
df.loc[(df['OFNS_DESC'] == 'FORGERY'),
       'CRIME'] = 'FORGERY'

df.loc[(df['OFNS_DESC'] == 'ADMINISTRATIVE CODE'),
       'CRIME'] = 'ADMINISTRATIVE CODE'

df.loc[(df['OFNS_DESC'] == 'ARSON'),
       'CRIME'] = 'ARSON'
       
df.loc[(df['OFNS_DESC'] == 'GAMBLING'),
       'CRIME'] = 'GAMBLING'

df.loc[(df['OFNS_DESC'] == 'VEHICLE AND TRAFFIC LAWS') | 
       (df['OFNS_DESC'] == 'INTOXICATED & IMPAIRED DRIVING')| 
        (df['OFNS_DESC'] == 'UNAUTHORIZED USE OF A VEHICLE'),
       'CRIME'] = 'VEHICLE AND TRAFFIC LAWS'

df.loc[(df['OFNS_DESC'] == 'PETIT LARCENY') | (df['OFNS_DESC'] == 'GRAND LARCENY') |
       (df['OFNS_DESC'] == 'ROBBERY') | 
       (df['OFNS_DESC'] == 'BURGLARY') | 
       (df['OFNS_DESC'] == "BURGLAR'S TOOLS") |
       (df['OFNS_DESC'] == 'THEFT OF SERVICES') |
       (df['OFNS_DESC'] == 'POSSESSION OF STOLEN PROPERTY') |
       (df['OFNS_DESC'] == 'OTHER OFFENSES RELATED TO THEF') |
       (df['OFNS_DESC'] == 'GRAND LARCENY OF MOTOR VEHICLE'),
       'CRIME'] = 'THEFT/LARCENY'

df.loc[(df['OFNS_DESC'] == 'ASSAULT 3 & RELATED OFFENSES') | (df['OFNS_DESC'] == 'FELONY ASSAULT') |
       (df['OFNS_DESC'] == 'OFFENSES AGAINST THE PERSON'),
       'CRIME'] = 'ASSAULT'

df.loc[(df['OFNS_DESC'] == 'HARRASSMENT 2'),
       'CRIME'] = 'HARRASSMENT'

df.loc[(df['OFNS_DESC'] == 'CRIMINAL MISCHIEF & RELATED OF'),
       'CRIME'] = 'CRIMINAL MISCHIEF'

df.loc[(df['OFNS_DESC'] == 'OFF. AGNST PUB ORD SENSBLTY &')|
       (df['OFNS_DESC'] == 'OFFENSES AGAINST PUBLIC ADMINI'),
       'CRIME'] = 'PUBLIC DISORDER'

df.loc[(df['CMPLNT_FR_DT'] == 'DANGEROUS DRUGS'),
       'CRIME'] = 'DANGEROUS DRUGS'

df.loc[(df['OFNS_DESC'] == 'THEFT-FRAUD') | (df['OFNS_DESC'] == 'FRAUDS') |
       (df['OFNS_DESC'] == 'OFFENSES INVOLVING FRAUD') | (df['OFNS_DESC'] == 'FRAUDULENT ACCOSTING') ,
       'CRIME'] = 'FRAUD'

df.loc[(df['OFNS_DESC'] == 'SEX CRIMES') | (df['OFNS_DESC'] == 'RAPE'),
       'CRIME'] = 'SEX CRIMES'

df.loc[(df['OFNS_DESC'] == 'DANGEROUS WEAPONS'),
       'CRIME'] = 'DANGEROUS WEAPONS'
       
df.loc[(df['OFNS_DESC'] == 'CRIMINAL TRESPASS'),
       'CRIME'] = 'CRIMINAL TRESPASS'
       
df.loc[(df['OFNS_DESC'] == 'FORGERY'),
       'CRIME'] = 'FORGERY'

df.loc[(df['OFNS_DESC'] == 'ADMINISTRATIVE CODE'),
       'CRIME'] = 'ADMINISTRATIVE CODE'

df.loc[(df['OFNS_DESC'] == 'ARSON'),
       'CRIME'] = 'ARSON'
       
df.loc[(df['OFNS_DESC'] == 'GAMBLING'),
       'CRIME'] = 'GAMBLING'

df.loc[(df['OFNS_DESC'] == 'VEHICLE AND TRAFFIC LAWS') | 
       (df['OFNS_DESC'] == 'INTOXICATED & IMPAIRED DRIVING')| 
        (df['OFNS_DESC'] == 'UNAUTHORIZED USE OF A VEHICLE'),
       'CRIME'] = 'VEHICLE AND TRAFFIC LAWS'

df['CRIME'].value_counts()

df.drop(df[df['OFNS_DESC'] == 'MISCELLANEOUS PENAL LAW'].index, axis=0, inplace=True)
df.drop(df[df['OFNS_DESC'] == 'NYS LAWS-UNCLASSIFIED FELONY'].index, axis=0, inplace=True)
df.drop(df[df['OFNS_DESC'] == 'OTHER STATE LAWS (NON PENAL LA'].index, axis=0, inplace=True)
df.drop(['OFNS_DESC', 'KY_CD'], axis=1, inplace=True)

# Categorizing VIC_SEX & VIC_AGE_GROUP
df['VIC_SEX'].value_counts()
df['VIC_AGE_GROUP'].value_counts()
df = df[df.groupby('VIC_AGE_GROUP')['VIC_AGE_GROUP'].transform('size') > 10000]
df.drop(df[df['VIC_AGE_GROUP'] == 'UNKNOWN'].index, axis=0, inplace=True)
df['VIC_AGE_GROUP'].value_counts()
df['VIC_SEX'].value_counts()
df.drop(df[df['VIC_SEX'] == 'D'].index, axis=0, inplace=True)
df.drop(df[df['VIC_SEX'] == 'E'].index, axis=0, inplace=True)
df['VIC_SEX'].value_counts()

# Categorizing BORO_NM
df['BORO_NM'].value_counts()
df.drop(['BORO_NM'], axis=1, inplace=True)

# Categorizing CMPLNT_FR_DT and CMPLNT_FR_TM 	CMPLNT_FR_TM_CODED
df.loc[df.CMPLNT_FR_DT.str.contains('^12'), 'Season'] = 'Winter'
df.loc[df.CMPLNT_FR_DT.str.contains('^1'), 'Season'] = 'Winter'
df.loc[df.CMPLNT_FR_DT.str.contains('^2'), 'Season'] = 'Winter'
df.loc[df.CMPLNT_FR_DT.str.contains('^3'), 'Season'] = 'Spring'
df.loc[df.CMPLNT_FR_DT.str.contains('^4'), 'Season'] = 'Spring'
df.loc[df.CMPLNT_FR_DT.str.contains('^5'), 'Season'] = 'Spring'
df.loc[df.CMPLNT_FR_DT.str.contains('^6'), 'Season'] = 'Summer'
df.loc[df.CMPLNT_FR_DT.str.contains('^7'), 'Season'] = 'Summer'
df.loc[df.CMPLNT_FR_DT.str.contains('^8'), 'Season'] = 'Summer'
df.loc[df.CMPLNT_FR_DT.str.contains('^9'), 'Season'] = 'Autumn'
df.loc[df.CMPLNT_FR_DT.str.contains('^10'), 'Season'] = 'Autumn'
df.loc[df.CMPLNT_FR_DT.str.contains('^11'), 'Season'] = 'Autumn'
df['Season'].value_counts()
df.loc[df.CMPLNT_FR_TM.str.startswith('6:'), 'Time_of_Day'] = 'Morning'
df.loc[df.CMPLNT_FR_TM.str.startswith('7:'), 'Time_of_Day'] = 'Morning'
df.loc[df.CMPLNT_FR_TM.str.startswith('8:'), 'Time_of_Day'] = 'Morning'
df.loc[df.CMPLNT_FR_TM.str.startswith('9:'), 'Time_of_Day'] = 'Morning'
df.loc[df.CMPLNT_FR_TM.str.startswith('10:'), 'Time_of_Day'] = 'Morning'
df.loc[df.CMPLNT_FR_TM.str.startswith('11:'), 'Time_of_Day'] = 'Morning'
df.loc[df.CMPLNT_FR_TM.str.startswith('12:'), 'Time_of_Day'] = 'AfterNoon'
df.loc[df.CMPLNT_FR_TM.str.startswith('13:'), 'Time_of_Day'] = 'AfterNoon'
df.loc[df.CMPLNT_FR_TM.str.startswith('14:'), 'Time_of_Day'] = 'AfterNoon'
df.loc[df.CMPLNT_FR_TM.str.startswith('15:'), 'Time_of_Day'] = 'AfterNoon'
df.loc[df.CMPLNT_FR_TM.str.startswith('16:'), 'Time_of_Day'] = 'AfterNoon'
df.loc[df.CMPLNT_FR_TM.str.startswith('17:'), 'Time_of_Day'] = 'AfterNoon'
df.loc[df.CMPLNT_FR_TM.str.startswith('18:'), 'Time_of_Day'] = 'Evening'
df.loc[df.CMPLNT_FR_TM.str.startswith('19:'), 'Time_of_Day'] = 'Evening'
df.loc[df.CMPLNT_FR_TM.str.startswith('20:'), 'Time_of_Day'] = 'Evening'
df.loc[df.CMPLNT_FR_TM.str.startswith('21:'), 'Time_of_Day'] = 'Evening'
df.loc[df.CMPLNT_FR_TM.str.startswith('22:'), 'Time_of_Day'] = 'Evening'
df.loc[df.CMPLNT_FR_TM.str.startswith('23:'), 'Time_of_Day'] = 'Evening'
df.loc[df.CMPLNT_FR_TM.str.startswith('0:'), 'Time_of_Day'] = 'Evening'
df.loc[df.CMPLNT_FR_TM.str.startswith('1:'), 'Time_of_Day'] = 'PastMidnight'
df.loc[df.CMPLNT_FR_TM.str.startswith('2:'), 'Time_of_Day'] = 'PastMidnight'
df.loc[df.CMPLNT_FR_TM.str.startswith('3:'), 'Time_of_Day'] = 'PastMidnight'
df.loc[df.CMPLNT_FR_TM.str.startswith('4:'), 'Time_of_Day'] = 'PastMidnight'
df.loc[df.CMPLNT_FR_TM.str.startswith('5:'), 'Time_of_Day'] = 'PastMidnight'

df['Time_of_Day'].value_counts()
df['LAW_CAT_CD'].value_counts()
df['isFELONY'].value_counts()
df.drop(['CMPLNT_FR_TM', 'CMPLNT_FR_DT', 'CMPLNT_FR_TM_CODED',  "isFELONY"], axis=1, inplace=True)
df.columns
df.shape

# Dummies and Final DF
dummies = pd.get_dummies(df[['Season', 'VIC_SEX', 'VIC_AGE_GROUP',
                            'Time_of_Day', 'PREM_TYP']], drop_first=True)
df_final = pd.merge(df, dummies, left_index=True, right_index=True)
df_final.drop(['LAW_CAT_CD', 'VIC_AGE_GROUP', 'VIC_SEX', 
               'Season', 'Time_of_Day', 'PREM_TYP', 'isSTATEN ISLAND',], axis=1, inplace=True)
df_final.head()
df_final.shape
df_final.describe()
df_final.columns
sns.heatmap(df_final.isna(), cbar=False)
df_final= df_final.dropna()

# MOdelling
df_final.columns

df_final.columns = ['isBRONX', 'isMANHATTAN', 'isBROOKLYN', 'isQUEENS', 'CRIME',
       'Season_Spring', 'Season_Summer', 'Season_Winter', 'VIC_SEX_M',
       'VIC_AGE_GROUP_25-44', 'VIC_AGE_GROUP_45-64', 'VIC_AGE_GROUP_65',
       'VIC_AGE_GROUP_less18', 'Time_of_Day_Evening', 'Time_of_Day_Morning',
       'Time_of_Day_PastMidnight', 'PREM_TYP_BANK', 'PREM_TYP_CEMETERY',
       'PREM_TYP_COMMERCIAL BUILDING', 'PREM_TYP_CONSTRUCTION SITE',
       'PREM_TYP_GROCERY SHOPPING', 'PREM_TYP_HOMELESS SHELTER',
       'PREM_TYP_HOUSE OF WORSHIP', 'PREM_TYP_MEDICAL FACILITY',
       'PREM_TYP_PUBLIC PARKING LOTS', 'PREM_TYP_PUBLIC.PRIVATE TRANSPORT',
       'PREM_TYP_RESIDENCE', 'PREM_TYP_RESTAURANT.BAR.HOTEL',
       'PREM_TYP_SCHOOL.PLAYGROUND', 'PREM_TYP_SHOPPING',
       'PREM_TYP_SPA.BEAUTYSALON.GYM', 'PREM_TYP_ST.Rd.Hw.Brdg']

# df_final.to_csv('df_final.csv')
# df_final = pd.read_csv('df_final.csv')
# df_final.drop(['Unnamed: 0'], axis=1, inplace=True)
y = df_final.loc[:,["CRIME"]]
X = df_final.drop(['CRIME'], axis=1)
y.isna().sum()
X.isna().sum()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

# XGBoost
XGBoost = xgb.XGBClassifier(tree_method='hist', max_delta_step=1, verbosity = 2)

v_xgb = XGBoost.fit(X_train, y_train)
y_hat_train_XG_v = v_xgb.predict(X_train)
y_hat_test_XG_v = v_xgb.predict(X_test)
training_accuracy = accuracy_score(y_train, y_hat_train_XG_v)
val_accuracy = accuracy_score(y_test, y_hat_test_XG_v)
print("Training Accuracy: {:.4}%".format(training_accuracy * 100))
print("Validation accuracy: {:.4}%".format(val_accuracy * 100))
print(v_xgb.get_xgb_params())

# Only run if you have time
# param_grid_XG = {
#     "learning_rate": [0.1,0.2,0.4],
#     'max_depth': [10,15,20],
#     'min_child_weight': [3,5,7],
#     'subsample': [0.5,0.7],
# }
# grid_clf = GridSearchCV(XGBoost, param_grid_XG, scoring='accuracy', cv=5, n_jobs=1, verbose=5)
# grid_clf.fit(X_train, y_train)
# best_parameters = grid_clf.best_params_

# print("Grid Search found the following optimal parameters: ")
# for param_name in sorted(best_parameters.keys()):
#     print("%s: %r" % (param_name, best_parameters[param_name]))

# y_hat_train_XG = grid_clf.predict(X_train)
# y_hat_test_XG = grid_clf.predict(X_test)
# training_accuracy = accuracy_score(y_train, y_hat_train_XG)
# val_accuracy = accuracy_score(y_test, y_hat_test_XG)

# print("")
# print("Training Accuracy: {:.4}%".format(training_accuracy * 100))
# print("Validation accuracy: {:.4}%".format(val_accuracy * 100))

# from sklearn.externals import joblib
# joblib.dump(grid_clf.best_estimator_, 'xgb_best_params.pkl')
# joblib.dump(grid_clf.best_estimator_, 'xgb_best_params_compress.pkl', compress = 1)

XGBoost_2 = xgb.XGBClassifier(verbosity = 2, learning_rate=0.01, max_depth=12, min_child_weight=6, subsample=0.3)
bestfit_xgb = XGBoost_2.fit(X_train, y_train)
y_hat_train_XG_best = bestfit_xgb.predict(X_train)
y_hat_test_XG_best = bestfit_xgb.predict(X_test)
training_accuracy2 = accuracy_score(y_train, y_hat_train_XG_best)
val_accuracy2 = accuracy_score(y_test, y_hat_test_XG_best)
print("Training Accuracy: {:.4}%".format(training_accuracy2 * 100))
print("Validation accuracy: {:.4}%".format(val_accuracy2 * 100))




# kNN
from collections import defaultdict
from scipy.stats import hmean
from scipy.spatial.distance import cdist
from scipy import stats
import numbers
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

kNN_v = KNeighborsClassifier()
kNN_v.fit(X_train, y_train)

y_hat_train_kNN_v = kNN_v.predict(X_train)
y_hat_test_kNN_v = kNN_v.predict(X_test)
training_accuracy_kNNv = accuracy_score(y_train, y_hat_train_kNN_v)
val_accuracy_kNNv = accuracy_score(y_test, y_hat_test_kNN_v)
print("Training Accuracy: {:.4}%".format(training_accuracy_kNNv * 100))
print("Validation accuracy: {:.4}%".format(val_accuracy_kNNv * 100))


# Only run if you have time
# param_grid_kNN = {
#     "n_neighbors": [3,5,7,9,11,13,15,17,19],
#     'weights': ['uniform', 'distance'],
#     'metric': ['euclidean', 'manhattan']
# }

# grid_clf_kNN = GridSearchCV(kNN_v, param_grid_kNN, scoring='accuracy', cv=5, n_jobs=1, verbose=5)
# grid_clf_kNN.fit(X_train, y_train)
# best_parameters_kNN = grid_clf_kNN.best_params_

# print("Grid Search found the following optimal parameters: ")
# for param_name in sorted(best_parameters_kNN.keys()):
#     print("%s: %r" % (param_name, best_parameters_kNN[param_name]))

# y_hat_train_kNN = grid_clf_kNN.predict(X_train)
# y_hat_test_kNN = grid_clf_kNN.predict(X_test)
# training_accuracy_kNNGS = accuracy_score(y_train, y_hat_train_kNN)
# val_accuracy_kNNGS = accuracy_score(y_test, y_hat_test_kNN)

# print("")
# print("Training Accuracy: {:.4}%".format(training_accuracy_kNNGS * 100))
# print("Validation accuracy: {:.4}%".format(val_accuracy_kNNGS * 100))

""" Result 
Grid Search found the following optimal parameters: 
metric: 'euclidean'
n_neighbors: 19
weights: 'uniform'

Training Accuracy: 37.2%
Validation accuracy: 34.91%
"""