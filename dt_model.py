
import numpy as np
import math
import pandas as pd

data = pd.read_csv("df_2019-2020_modified.csv")
print("imported")
data.loc[data["LAW_CAT_CD"] == "FELONY", "isFELONY"] = 1 #filters and applys
data.loc[data["LAW_CAT_CD"] != "FELONY", "isFELONY"] = 0 #filters and applys

#data.to_csv("felony.csv")

"""
This section encodes the borough in their own variable, as a feature, so we can numerically
check for each one in the decision tree.

Not coding to Bronx = 1, Manhattan = 2 because that would introduce ordinality which the boro
feature should not have.
"""
boroughs = data["BORO_NM"].unique()
boroughFeatures = []
for borough in boroughs:
    try:
        if (math.isnan(borough)): borough = "nan"
    except: pass
    data.loc[data["BORO_NM"] == borough, "is" + borough] = 1 #filters and applys
    data.loc[data["BORO_NM"] != borough, "is" + borough] = 0 #filters and applys
    boroughFeatures.append("is" + borough)

"""
SUSP_AGE_GROUP
make the variable 1-4, which is ok to be ordinal because it is age, and replace na or malformed values with the average
"""
#ages = data["SUSP_AGE_GROUP"].unique()
ageDict = {1:"<18", 2:"18-24", 3:"45-64", 4:"65+"}
ageTotal = 0
ageNumber = 0
for code in ageDict:
    age = ageDict[code]
    data.loc[data["SUSP_AGE_GROUP"] == age, "SUSP_AGE_GROUP_CODED"] = code #filters and applys
    n = data[data["SUSP_AGE_GROUP_CODED"] == code].shape[0]
    ageNumber +=  n
    ageTotal += code*n
data["SUSP_AGE_GROUP_CODED"] = data["SUSP_AGE_GROUP_CODED"].fillna(ageTotal/ageNumber)
data.to_csv("ageaverage.csv")

"""
Compile up the features for the model
"""
features = ['CMPLNT_FR_DT_CODED',
            'CMPLNT_FR_TM_CODED',
            "SUSP_AGE_GROUP_CODED"]#,
           # 'LOC_OF_OCCUR_DESC',
           # 'JURIS_DESC',
           # 'SUSP_RACE',
           # 'SUSP_SEX',
           # 'VIC_AGE_GROUP',
           # 'VIC_RACE',
           # 'VIC_SEX'] #column labels for features
features = features + boroughFeatures

y = data["isFELONY"].copy() #this is our target

# Q6: Copying the values from the clean_data dataset to new dataset x which only consist of the Morning Feature Data
# Hint: use a copy command in pandas
# %%------------------------------------------------------------------------------------------------------------

X = data[features].copy() #new dataset X as specified

# Q7: Perform Test and Train split . USe 20 percent for test set and use random state of 1
# %%------------------------------------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20, random_state=1) #perform the train-test split, as usual

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(max_leaf_nodes=50, random_state=0) #initialize the classifier for training
classifier.fit(X_train, y_train) #fit the model

predict = classifier.predict(X_test) #generate preditions

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

accuracy = accuracy_score(y_true = y_test, y_pred = predict)
print("Accuracy: " + str(accuracy))
f1 = f1_score(y_true = y_test, y_pred = predict)
print("F1: " + str(f1))
confusion = confusion_matrix(y_true = y_test, y_pred = predict)
print("Confusion Matrix:")
print(confusion)
#
# classifier = DecisionTreeClassifier(max_leaf_nodes=36, random_state=0) #initialize the classifier for training
# classifier.fit(X_train, y_train) #fit the model
#
# predict = classifier.predict(X_test) #generate preditions
# print(predict[:15]) #take a look at the values
# print(y_test[:15])
#
# accuracy = accuracy_score(y_true = y_test, y_pred = predict)
# print(accuracy)
# f1 = f1_score(y_true = y_test, y_pred = predict)
# print(f1)
# confusion = confusion_matrix(y_true = y_test, y_pred = predict)
# print(confusion)
