import sys
import pkg_resources
import subprocess

"""
Import, or import then install relevant packages
"""

try: import numpy as np
except:
    subprocess.check_call([sys.executable], "-m", "pip", "install", "numpy")
    import numpy as np
try: import pandas as pd
except:
    subprocess.check_call([sys.executable], "-m", "pip", "install", "pandas")
    import pandas as pd
try: import pickle
except:
    subprocess.check_call([sys.executable], "-m", "pip", "install", "pickle")
    import pickle
try:
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import confusion_matrix
except:
    subprocess.check_call([sys.executable], "-m", "pip", "install", "scikit-learn")
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import confusion_matrix

data = pd.read_csv("df_2019-2020_modified.csv")
print("imported")

#col must be cleaned to only contain wanted values or null
def populateRandDist(data, col):
    distribution = data[col].value_counts(normalize=True)
    nulls = data[col].isnull()
    data.loc[nulls, col] = np.random.choice(distribution.index, size=len(data[nulls]), p=distribution.values)
    return data

"""
Prep Targets
"""
print("performing some cleaning and encoding... please wait")
data.loc[data["LAW_CAT_CD"] == "FELONY", "isFELONY"] = 1 #filters and applys
data.loc[data["LAW_CAT_CD"] != "FELONY", "isFELONY"] = 0 #filters and applys

data.loc[data["LAW_CAT_CD"] == "MISDEMEANOR", "isMISDEMEANOR"] = 1 #filters and applys
data.loc[data["LAW_CAT_CD"] != "MISDEMEANOR", "isMISDEMEANOR"] = 0 #filters and applys

data.loc[data["LAW_CAT_CD"] == "VIOLATION", "isVIOLATION"] = 1 #filters and applys
data.loc[data["LAW_CAT_CD"] != "VIOLATION", "isVIOLATION"] = 0 #filters and applys
#data.to_csv("felony.csv")

"""
This section encodes the borough in their own variable, as a feature, so we can numerically
check for each one in the decision tree.

Not coding to Bronx = 1, Manhattan = 2 because that would introduce ordinality which the boro
feature should not have.
"""

data = populateRandDist(data, "BORO_NM")
boroughs = data["BORO_NM"].unique()
boroughFeatures = []
for borough in boroughs:
    data.loc[data["BORO_NM"] == borough, "is" + borough] = 1 #filters and applys
    data.loc[data["BORO_NM"] != borough, "is" + borough] = 0 #filters and applys
    boroughFeatures.append("is" + borough)

"""
race cleaning. There is some commented out code for suspect features, which is removed because we ended up not using suspect
information for the features in any model.
"""
# data["SUSP_RACE"] = data["SUSP_RACE"].fillna("UNKNOWN")
# races = data["SUSP_RACE"].unique()
# for race in races:
#     data.loc[data["SUSP_RACE"] == race, "isSUSP" + race] = 1 #filters and applys
#     data.loc[data["SUSP_RACE"] != race, "isSUSP" + race] = 0 #filters and applys
#     raceFeatures.append("isSUSP" + race)

raceFeatures = []
data["VIC_RACE"] = data["VIC_RACE"].fillna("UNKNOWN")
races = data["VIC_RACE"].unique()
for race in races:
    data.loc[data["VIC_RACE"] == race, "isVIC" + race] = 1 #filters and applys
    data.loc[data["VIC_RACE"] != race, "isVIC" + race] = 0 #filters and applys
    raceFeatures.append("isVIC" + race)

"""
SUSP_AGE_GROUP
make the variable 1-4, which is ok to be ordinal because it is age, and replace na or malformed values with the average
"""
#ages = data["SUSP_AGE_GROUP"].unique()
# ageTotal = 0
# ageNumber = 0
# for code in ageDict:
#     age = ageDict[code]
#     data.loc[data["SUSP_AGE_GROUP"] == age, "SUSP_AGE_GROUP_CODED"] = code #filters and applys
#     n = data[data["SUSP_AGE_GROUP_CODED"] == code].shape[0]
#     ageNumber +=  n
#     ageTotal += code*n
# data["SUSP_AGE_GROUP_CODED"] = data["SUSP_AGE_GROUP_CODED"].fillna(ageTotal/ageNumber)
ageDict = {1:"<18", 2:"18-24", 3:"45-64", 4:"65+"}
ageTotal = 0
ageNumber = 0
for code in ageDict:
    age = ageDict[code]
    data.loc[data["VIC_AGE_GROUP"] == age, "VIC_AGE_GROUP_CODED"] = code #filters and applys
    n = data[data["VIC_AGE_GROUP_CODED"] == code].shape[0]
    ageNumber +=  n
    ageTotal += code*n
data["VIC_AGE_GROUP_CODED"] = data["VIC_AGE_GROUP_CODED"].fillna(ageTotal/ageNumber)


"""
Clean out the misc. sex labels, set them to null, and replace them with a random distribution based on
how many M/F there are
"""
# notM = data["SUSP_SEX"] != "M"
# notF = data["SUSP_SEX"] != "F"
# data.loc[notM & notF, "SUSP_SEX"] = np.nan
# data.loc[data["SUSP_SEX"] == "U", "SUSP_SEX"] = np.nan
# data = populateRandDist(data, "SUSP_SEX")
# # distribution = data["SUSP_SEX"].value_counts(normalize = True)
# # nulls = data["SUSP_SEX"].isnull()
# # data.loc[nulls,"SUSP_SEX"] = np.random.choice(distribution.index, size = len(data[nulls]), p = distribution.values)
# notM = data["SUSP_SEX"] != "M"
# notF = data["SUSP_SEX"] != "F"
# data.loc[notM & notF, "SUSP_SEX"] = np.nan
# data.loc[data["SUSP_SEX"] == "U", "SUSP_SEX"] = np.nan
# data = populateRandDist(data, "SUSP_SEX")


# for sex in sexes:
#     data.loc[data["SUSP_SEX"] == sex, "isSUSP" + sex] = 1 #filters and applys
#     data.loc[data["SUSP_SEX"] != sex, "isSUSP" + sex] = 0 #filters and applys
#     sexFeatures.append("isSUSP" + sex)
# # distribution = data["SUSP_SEX"].value_counts(normalize = True)
# # nulls = data["SUSP_SEX"].isnull()
# # data.loc[nulls,"SUSP_SEX"] = np.random.choice(distribution.index, size = len(data[nulls]), p = distribution.values)
sexes = ["M", "F"]
sexFeatures = []
notM = data["VIC_SEX"] != "M"
notF = data["VIC_SEX"] != "F"
data.loc[notM & notF, "VIC_SEX"] = np.nan
data.loc[data["VIC_SEX"] == "U", "VIC_SEX"] = np.nan
data = populateRandDist(data, "VIC_SEX")

for sex in sexes:
    data.loc[data["VIC_SEX"] == sex, "isVIC" + sex] = 1 #filters and applys
    data.loc[data["VIC_SEX"] != sex, "isVIC" + sex] = 0 #filters and applys
    sexFeatures.append("isVIC" + sex)

"""
LOC_OF_OCCUR_DESC
"""
data = populateRandDist(data, "LOC_OF_OCCUR_DESC")
# distribution = data["LOC_OF_OCCUR_DESC"].value_counts(normalize = True)
# nulls = data["LOC_OF_OCCUR_DESC"].isnull()
# data.loc[nulls,"LOC_OF_OCCUR_DESC"] = np.random.choice(distribution.index, size = len(data[nulls]), p = distribution.values)

locations = data["LOC_OF_OCCUR_DESC"].unique()
locFeatures = []
for loc in locations:
    data.loc[data["LOC_OF_OCCUR_DESC"] == loc, "is" + loc] = 1 #filters and applys
    data.loc[data["LOC_OF_OCCUR_DESC"] != loc, "is" + loc] = 0 #filters and applys
    locFeatures.append("is" + loc)

"""
JURIS_DESC
"""
juris = data["JURIS_DESC"].unique()
jurisFeatures = []
for jur in juris:
    data.loc[data["JURIS_DESC"] == jur, "is" + jur] = 1 #filters and applys
    data.loc[data["JURIS_DESC"] != jur, "is" + jur] = 0 #filters and applys
    jurisFeatures.append("is" + jur)

"""
PREM_TYP_DESC
"""
data = data[data['PREM_TYP_DESC'].notna()]
prems = data["PREM_TYP_DESC"].unique()
premFeatures = []
for prem in prems:
    data.loc[data["PREM_TYP_DESC"] == prem, "is" + prem] = 1 #filters and applys
    data.loc[data["PREM_TYP_DESC"] != prem, "is" + prem] = 0 #filters and applys
    premFeatures.append("is" + prem)

print("Exporting cleaned data...")
data.to_csv("data_dt.csv")

print("Generating models...")

def scoring(y_test, predict):
    accuracy = accuracy_score(y_true=y_test, y_pred=predict)
    print("Accuracy: " + str(accuracy))
    f1 = f1_score(y_true=y_test, y_pred=predict)
    print("F1: " + str(f1))
    confusion = confusion_matrix(y_true=y_test, y_pred=predict)
    print("Confusion Matrix:")
    print(confusion)

def model(target, features):

    y = data[target].copy() #this is our target
    X = data[features].copy() #new dataset X as specified

    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = .10)# random_state=1) #perform the train-test split, as usual

    classifier = DecisionTreeClassifier(max_leaf_nodes = 50, random_state = 0) #initialize the classifier for training
    classifier.fit(X_train, y_train) #fit the model

    predict = classifier.predict(X_test) #generate preditions
    predictDF = pd.DataFrame(predict, columns = ["predict"])
    #predictDF["ytest"] = y_test
    predictDF.to_csv(target + "predictions_v2.csv")
    scoring(y_test, predict)

    pickle.dump(classifier, open("dt_model" + target + ".sav", "wb"))

    outputs = {"X_train":X_train, "X_test":X_test, "y_train":y_train, "y_test":y_test}
    for output in outputs: outputs[output].to_csv(target + "_" + output + ".csv")

"""
Compile up the features for the model
"""
features = ["CMPLNT_FR_DT_CODED",
            "CMPLNT_FR_TM_CODED",
            "VIC_AGE_GROUP_CODED"] #column labels for features

#add all the features we've built up
features = features + boroughFeatures + raceFeatures + sexFeatures + locFeatures + jurisFeatures + premFeatures

model("isFELONY", features)
model("isMISDEMEANOR", features)
model("isVIOLATION", features)

# def model(target, features):
#
#     y = data["isFELONY"].copy() #this is our target
#     y2 = data["isMISDEMEANOR"].copy()
#
#     X = data[features].copy() #new dataset X as specified

#     X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.10)# random_state=1) #perform the train-test split, as usual
#     X_train2,X_test2,y_train2,y_test2 = train_test_split(X,y2,test_size=.10)# random_state=1) #perform the train-test split, as usual
#
#
#
#     classifier = DecisionTreeClassifier(max_leaf_nodes=50, random_state=0) #initialize the classifier for training
#     classifier.fit(X_train, y_train) #fit the model
#
#     predict = classifier.predict(X_test) #generate preditions
#
#
#     scoring(y_test, predict)
#     classifier = DecisionTreeClassifier(max_leaf_nodes=50, random_state=0) #initialize the classifier for training
#     classifier.fit(X_train2, y_train2) #fit the model
#
#     predict = classifier.predict(X_test2) #generate preditions
#
#     scoring(y_test, predict)
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
