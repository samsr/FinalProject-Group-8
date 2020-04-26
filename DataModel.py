

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE
from sklearn.svm import SVC

df = pd.read_csv('cleaned.csv', index_col='Unnamed: 0', nrows=50000)
print(df.head())


########## RANDOM FOREST ##########

X = df.iloc[:,3:-4]
Y = df.iloc[:,-2]

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.25, random_state=13)
rfc = RandomForestClassifier(bootstrap=True, n_estimators=100)
rfc.fit(x_train,y_train)
y_pred = rfc.predict(x_test)
predict_prob = rfc.predict_proba(x_test)

importances = rfc.feature_importances_
imp = pd.Series(importances,X.columns)

#importance = ['premis_var_1', 'BORO1', 'BORO3', 'BORO5', 'VIC_AGE1', 'VIC_AGE3', 'VIC_AGE4', 'VIC_AGE5',
#               'LOC_DESC1', 'LOC_DESC2', 'LOC_DESC3', 'LOC_DESC4', 'VIC_RACE1', 'VICM_SEX1', 'VICM_SEX2']

def imp_vars(importance):
    # make the bar Plot from f_importances
    importance.plot(x='Features', y='Importance', kind='bar')
    plt.show()


def class_rpt(yt,yp):
    print("Clasification Report:")
    print(classification_report(yt,yp))
    print()
    print("Confusion Matrix:")
    print(confusion_matrix(yt, yp))
    print("Accuracy:", accuracy_score(yt,yp))
    print()


imp_vars(imp)
class_rpt(y_test,y_pred)


########## LOGISTIC REGRESSION ##########

X_logr = df.iloc[:,3:-3]
Y_logr = df.iloc[:,-2]

lr = LogisticRegression(solver='lbfgs')
rfe = RFE(lr)
rfe.fit(X_logr,Y_logr)
print(rfe.support_)
print(rfe.ranking_)
print(X.columns[rfe.support_])
#importance = ['premis_var_1', 'premis_var_2', 'premis_var_3', 'premis_var_4', 'BORO1', 'BORO4', 'VIC_AGE1',
#              'VIC_AGE2', 'VIC_AGE3', 'VIC_AGE5', 'LOC_DESC1', 'LOC_DESC2', 'VIC_RACE3', 'VICM_SEX1', 'VICM_SEX2']

xl_train, xl_test, yl_train, yl_test = train_test_split(X_logr,Y_logr,test_size=0.25, random_state=13)
lr = LogisticRegression(solver='lbfgs')
lr.fit(xl_train,yl_train)
y_pred_lr = lr.predict(xl_test)

class_rpt(yl_test,y_pred_lr)

########## SUPPORT VECTOR MACHINE ##########

clf = SVC(kernel='linear')
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

print("Classification Report:\n")
print(classification_report(y_test,y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred))


'''
print("ROC_AUC:", roc_auc_score(y_test,predict_prob[:,1]))
print()


base_fpr, base_tpr, _ = roc_curve(y_test, [1 for _ in range(len(y_test))])
model_fpr, model_tpr, _ = roc_curve(y_test, predict_prob[:,1])

# Plot both curves
plt.plot(base_fpr, base_tpr, 'b', label='baseline')
plt.plot(model_fpr, model_tpr, 'r', label='model')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.show()
'''