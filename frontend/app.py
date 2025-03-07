import streamlit as st
import requests
import pandas as pd

st.title("Customer Segmentation")

st.write("Enter customer details below to predict their segment.")

# Input form
recency = st.number_input("Recency (days since last purchase)", min_value=0)
frequency = st.number_input("Frequency (number of purchases)", min_value=0)
monetary = st.number_input("Monetary (total spend)", min_value=0.0)
total_returns = st.number_input("Total Returns", min_value=0)

if st.button("Predict Segment"):
    data = {
        "Recency": recency,
        "Frequency": frequency,
        "Monetary": monetary,
        "Total_Returns": total_returns
    }
    response = requests.post("http://127.0.0.1:5000/predict", json=data)
    
    if response.status_code == 200:
        result = response.json()
        cluster = result['cluster']

        # Display results with description
        st.success(f"Predicted Cluster: {cluster}")

        if cluster == 0:
            st.info("**Cluster 0:** High-value frequent customers. Target with loyalty programs.")
        elif cluster == 1:
            st.info("**Cluster 1:** New or infrequent customers. Focus on re-engagement strategies.")
        elif cluster == 2:
            st.info("**Cluster 2:** Customers at risk of churn. Offer discounts or promotions.")
        elif cluster == 3:
            st.info("**Cluster 3:** Budget-conscious customers. Provide value-based offers.")
    else:
        st.error("Error in prediction.")
