import streamlit as st
import joblib
import pandas as pd

model = joblib.load("Customer_churn_RandomForest.pkl")

st.title("📊 Customer Churn Prediction")
st.markdown("Predict whether a customer is likely to churn using a Random Forest model.")
st.sidebar.header("About")
st.sidebar.info(
    """
    Machine Learning Model:
    - Random Forest
    - Hyperparameter Tuned
    - Built with Scikit-Learn
    - Deployed using Streamlit
    """
)

tenure = st.number_input("Tenure")
monthly = st.number_input("Monthly Charges")
total = st.number_input("Total Charges")
churn_score = st.number_input("Churn Score")
cltv = st.number_input("CLTV")

if st.button("Predict"):
    data = pd.DataFrame([{'Tenure Months': tenure,'Monthly Charges': monthly,'Total Charges': total,'Churn Score': churn_score,'CLTV': cltv}])
	
    prediction = model.predict(data)
    probability = model.predict_proba(data)
	
    if prediction[0] == 1:
        st.error("Customer is likely to Churn")
    else:
        st.success("Customer is likely to Stay")

    st.metric("Churn Probability:",round(probability[0][1] * 100, 2),"%")
    risk = probability[0][1]

    if risk < 0.3:
        st.success("🟢 Low Risk")
    elif risk < 0.7:
        st.warning("🟡 Medium Risk")
    else:
       	st.error("🔴 High Risk")