import streamlit as st

import numpy as np 

from sklearn.preprocessing import StandardScaler

import joblib



st.set_page_config(layout="wide")


scaler = StandardScaler()


st.title("Restaurant Rating Prediction App")


st.set_page_config(layout="wide")


scaler = joblib.load("scaler.pkl")



st.caption("This app helps you to predict a restaurant review class")


st.divider()


averagecost = st.number_input("Please enter the estimated average cost for two", min_value=50, max_value=999999, value=1000, step=200)


tablebooking = st.selectbox("Does the restaurant accept table booking?", options=["Yes", "No"])


onlinedelivery = st.selectbox("Does the restaurant accept online delivery?", options=["Yes", "No"])

pricerange = st.selectbox("What is the price range of the restaurant? (1 Cheapest, 4 Most Expensive)", options=[1, 2, 3, 4])


predictionbutton = st.button("Predict Review Class")


st.divider()


joblib.load("model.pkl")


model = joblib.load("model.pkl")


bookingstatus = 1 if tablebooking == "Yes" else 0


deliverystatus = 1 if onlinedelivery == "Yes" else 0


values = np.array([[averagecost, bookingstatus, deliverystatus, pricerange]])

my_X_values = np.array(values)


X = scaler.transform(my_X_values)





if predictionbutton:
    st.snow()

    prediction = model.predict(X)

    # Above 2 below 2.5   poor
    # Above 2.5 below 3.5 average
    # Above 3.5 below 4.0 good
    # Above 4.0 below 4.5 very good
    # Above 4.5 below 5.0 excellent

    if prediction < 2.5:
        st.write("Poor")
    elif prediction < 3.5:
        st.write("Average")
    elif prediction < 4.0:
        st.write("Good")
    elif prediction < 4.5:
        st.write("Very Good")
    else:
        st.write("Excellent")

    


