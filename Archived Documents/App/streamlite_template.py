## streamlite template
## use line below to install streamlit
## pip install streamlt
## Documentation is below
## https://www.streamlit.io/
## run app in command line: 
## streamlit run your_script.py

## basic libraries for processing data and plotting graphs
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import plotly.express as px
import streamlit as st
from io import BytesIO
import random
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import time
import itertools
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")

##  load dataframe using pickle
@st.cache(persist=True)
def explore_data(df):
    df = pickle.load(open('pickle/'+df+'.p', 'rb'))
    return df

df_final  = explore_data('df_final')[0:5000]
df = explore_data('df_plt')[0:5000]
y_defined =  explore_data('y_defined')

## Header image
image = Image.open('images/MG_1_1_New_York_City-1.jpg')
st.image(image, caption='I <3 NYC', use_column_width=False)
st.title('NYC Crime')

# Display dataframe
st.dataframe(df)


# plotting matplotlib/sns graphs
fig1 = sns.distplot(df['KY_CD'])
st.write(fig1)
st.pyplot()

# st.markdown('\n')

fig2 = sns.pairplot(df)
st.write(fig2)
st.pyplot()

## Model Results
y = explore_data('y')
y_test = explore_data('y_test')
y_hat_test_rf = explore_data('y_hat_test_rf')

## Confusion Matrix Plot
normalize = True
cm = confusion_matrix(y_test, y_hat_test_rf)
plt.figure(figsize=(15,15))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()
tick_marks = np.arange(len(set(y)))
plt.xticks(tick_marks, set(y), rotation=45)
plt.yticks(tick_marks, set(y))
fmt = '.2f' if normalize else 'd'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
st.pyplot()


## Dropdown windows, sliders, etc...

st.write('''Enter your the information below:''')

box1 = st.selectbox("Select age", options = list(set(df['VIC_AGE_GROUP'])))
st.write('You selected:', box1)
X_b1 = pd.DataFrame({'VIC_AGE_GROUP_25-44':(1 if box1=='25-44' else 0), 'VIC_AGE_GROUP_45-64':(1 if box1=='45-64' else 0),'VIC_AGE_GROUP_65+':(1 if box1=='65+' else 0),'VIC_AGE_GROUP_<18':(1 if box1=='<18' else 0),'VIC_AGE_GROUP_UNKNOWN':(1 if box1=='UNKNOWN' else 0)}, index=[0])

st.markdown('\n')

box2 = st.selectbox("Select sex", list(set(df['VIC_SEX'])))
st.write('You selected:', box2)
X_b2 = pd.DataFrame({'VIC_SEX_E':(1 if box2=='E' else 0),'VIC_SEX_F':(1 if box2=='F' else 0),'VIC_SEX_M':(1 if box2=='M' else 0)}, index=[0])
st.markdown('\n')

box3 = st.selectbox('Select Borough', ['BORO_NM_BROOKLYN','BORO_NM_MANHATTAN','BORO_NM_QUEENS', 'BORO_NM_STATEN ISLAND'])
X_b3 = pd.DataFrame({'BORO_NM_BROOKLYN':(1 if box3=='BROOKLYN' else 0),'BORO_NM_MANHATTAN':(1 if box3=='MANHATTAN' else 0),'BORO_NM_QUEENS':(1 if box3=='QUEENS' else 0), 'BORO_NM_STATEN ISLAND':(1 if box3=='STATEN ISLAND' else 0)}, index=[0])
st.write('You selected:', box3)
st.markdown('\n')

slider1 = st.slider('Select KY_CD',min(df['KY_CD']), max(df['KY_CD']))
X_s1 = pd.DataFrame({'KY_CD':slider1}, index=[0])
st.write('You selected:', slider1)
st.markdown('\n')

X_1 = pd.concat([X_b1, X_b2, X_b3, X_s1], axis=1)

loaded_model = pickle.load(open('models/rf_model.sav', 'rb'))
result = loaded_model.predict(X_1)
crime = y_defined[(y_defined['n']==result[0])]['OFNS_DESC']
st.write(crime)
## For the template

## plotting 3D graphs
# fig = px.scatter_3d(df_plot, x='Dimension 1', y='Dimension 2', z='Dimension 3', color='labels',  title="""Three Component SVD Transformed Plot of Car Recommender Features Clustered with KAlgg""", height=1000, width=1000 ,hover_data=df_plot[['index']])
# st.write(fig)

# st.markdown('\n')

## plotting matplotlib/sns graphs
# fig1 = sns.scatterplot(x="x", y="y", data=df)
# st.write(fig1)



## Table using pandas
# st.write(df[(df['age']==option1) & (df['sex']==option2)][['age','sex','crine']])

# st.markdown('\n')
