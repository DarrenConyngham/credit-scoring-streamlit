import streamlit as st
import pandas as pd
import pickle
import sklearn
import numpy as np
import lightgbm

# amount = 200.0 
# income = 5221.29
# age = 22
# amount_previous_loans = 3790.0
# loan_duration = 60

# create a input data frame in streamlit
st.title("Loan Prediction")
st.subheader("Input Data")

# make streamlit wait for input

st.write("Please enter the following details to predict the loan approval percentage:")
amount = st.number_input("Loan Amount in Euros", value=100)
income = st.number_input("Income in Euros", value=100)
age = st.number_input("Age in Years", value=100)
amount_previous_loans = st.number_input("Amount Previous Loans in Euros", value=100)
loan_duration = st.number_input("Loan Duration in Months", value=100)

filename = 'loan_prediction_model_v2.pkl'

with open(filename, 'rb') as file:
    loaded_model = pickle.load(file)

single_data_list = [amount, income, age, amount_previous_loans, loan_duration]

prediction_array = np.array(single_data_list).reshape(1, -1)

percentage_prediction_of_default  = loaded_model.predict_proba(prediction_array)[0][1]

st.write("The predicted loan default percentage is: ", percentage_prediction_of_default * 100, "%")