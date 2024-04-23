import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

# Reading the Dataset
data = pd.read_csv('Faults.csv')

# Since the labels are in one hot encoding we will convert them into class labels
label_column = data.columns.values[-7:]
targets = data.iloc[:, -7:].idxmax(1)
dataset = data.drop(label_column, axis=1)
dataset['target'] = targets

# Correlation coeff of "TypeOfSteel_A300" and "TypeOfSteel_A400" is -1 so we can drop one of them.
# Correlation coeff of "X_Minimum" and "X_Maximum" is 1 so we can drop one of them.
# Similarly "Y_Minimum" and "Y_Maximum" is 1 so we can drop one of them.
dataset = dataset.drop(['TypeOfSteel_A400','X_Maximum','Y_Maximum'], axis=1)

# As we can see that importance of "SigmoidOfAreas" is very low we can remove that
dataset = dataset.drop('SigmoidOfAreas', axis=1)

# Split train test data
x, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]

# The features in the data set have different value ranges scaling needs to be done
x = StandardScaler().fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

# Train the Random Forest model
rf = RandomForestClassifier(random_state=0, n_estimators=100, min_samples_split=2)
rf.fit(x_train, y_train)

# Streamlit app

st.title('Fault Prediction')
# Dropdown to select row number
selected_row = st.selectbox('Select a row number:', options=list(range(len(data))))
# Display selected row
st.subheader('Selected Row:')
st.write(data.iloc[selected_row])
# Make prediction
prediction = rf.predict(x[selected_row:selected_row+1])
actual_target = dataset.iloc[selected_row]['target']
# Display prediction and actual target
st.subheader('Prediction:')
st.write(prediction[0])
st.subheader('Actual Target:')
st.write(actual_target)
