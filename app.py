import streamlit as st 
import numpy as np
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd 
import pickle

model = tf.keras.models.load_model('classification_model.keras')

#load the encoder and scalar
with open('preprocessing_files_experiment/label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('preprocessing_files_experiment/onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('preprocessing_files_experiment/scalar.pkl','rb') as file:
    scalar = pickle.load(file)


## streamlit app
st.title('Customer Churn Prediction',text_alignment='center')

#user input
geography = st.selectbox('Geography',onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age',18,95)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit_score')
estimated_salary = st.number_input('Estimated_salary')
tenure  = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is active Member',[0,1])

#prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

## one-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

#combine one hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

# Scale the input data
input_data_scaled = scalar.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.title(f'Churn Probability:{prediction_proba}')

if prediction_proba > 0.5:
    st.title('The customer is likely to churn.')
else:
    st.title('The customer is not likely to churn.')

