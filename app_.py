import streamlit as st
import pandas as pd
import pickle
import time

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ------------------------------
# LOAD FILES
# ------------------------------
model = pickle.load(open("model.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))

data = pd.read_csv("data.csv")

# ------------------------------
# CUSTOM CSS
# ------------------------------
st.markdown("""
<style>

.main-title{
    font-size:42px;
    font-weight:bold;
    color:#1E88E5;
}

.subtitle{
    font-size:18px;
    color:gray;
}

.card{
    padding:18px;
    border-radius:12px;
    background:#f7f7f7;
    box-shadow:0px 0px 10px rgba(0,0,0,0.12);
    margin-bottom:15px;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:50px;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------
# SIDEBAR
# ------------------------------

st.sidebar.title("📊 Customer Churn")

st.sidebar.markdown("---")

st.sidebar.success("Machine Learning Project")

st.sidebar.markdown("### Model")

st.sidebar.info("Logistic Regression")

st.sidebar.markdown("### Accuracy")

st.sidebar.success("81.69 %")

st.sidebar.markdown("### Dataset")

st.sidebar.info("7043 Customers")

st.sidebar.markdown("### Features")

st.sidebar.info("20 Features")

st.sidebar.markdown("---")

st.sidebar.write("👨‍💻 Developed by")

st.sidebar.success("Raj")

# ------------------------------
# TITLE
# ------------------------------

st.markdown(
    "<div class='main-title'>📊 Customer Churn Prediction Dashboard</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Predict whether a customer will churn using Machine Learning.</div>",
    unsafe_allow_html=True
)

st.divider()

# ------------------------------
# METRICS
# ------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Accuracy", "81.69%")

with col2:
    st.metric("Customers", "7043")

with col3:
    st.metric("Model", "Logistic Regression")

st.divider()

customer_id = st.text_input(
    "🔍 Enter Customer ID",
    placeholder="Example : 7590-VHVEG"
)

predict = st.button("🚀 Predict Customer Churn", use_container_width=True)
if predict:

    if customer_id.strip() == "":
        st.warning("⚠️ Please enter a Customer ID.")
        st.stop()

    with st.spinner("🔍 Searching customer and predicting..."):
        time.sleep(1)

        customer = data[data["customerID"] == customer_id]

        if customer.empty:
            st.error("❌ Customer ID not found.")
            st.info("Example Customer ID : 7590-VHVEG")
            st.stop()

        customer_display = customer.copy()

        st.divider()

        st.subheader("👤 Customer Information")

        col1, col2 = st.columns(2)

        with col1:
            st.info(f"**Gender:** {customer_display.iloc[0]['gender']}")
            st.info(f"**Senior Citizen:** {customer_display.iloc[0]['SeniorCitizen']}")
            st.info(f"**Partner:** {customer_display.iloc[0]['Partner']}")
            st.info(f"**Dependents:** {customer_display.iloc[0]['Dependents']}")
            st.info(f"**Tenure:** {customer_display.iloc[0]['tenure']} Months")

        with col2:
            st.info(f"**Internet Service:** {customer_display.iloc[0]['InternetService']}")
            st.info(f"**Contract:** {customer_display.iloc[0]['Contract']}")
            st.info(f"**Payment Method:** {customer_display.iloc[0]['PaymentMethod']}")
            st.info(f"**Monthly Charges:** ₹{customer_display.iloc[0]['MonthlyCharges']}")
            st.info(f"**Total Charges:** ₹{customer_display.iloc[0]['TotalCharges']}")

        prediction_data = customer.copy()

        prediction_data["TotalCharges"] = pd.to_numeric(
            prediction_data["TotalCharges"],
            errors="coerce"
        )

        prediction_data["TotalCharges"] = prediction_data["TotalCharges"].fillna(
            prediction_data["TotalCharges"].mean()
        )

        actual_value = prediction_data.iloc[0]["Churn"]

        prediction_data = prediction_data.drop(
            columns=["customerID", "Churn"]
        )

        for column in prediction_data.columns:

            if column in label_encoders:

                prediction_data[column] = label_encoders[column].transform(
                    prediction_data[column]
                )

        prediction = model.predict(prediction_data)[0]

        probability = model.predict_proba(prediction_data)[0]

        stay_probability = probability[0]
        churn_probability = probability[1]

        st.divider()

        st.subheader("🎯 Prediction Result")

        if prediction == 1:

            st.error("🔴 Customer is likely to Churn")

        else:

            st.success("🟢 Customer is likely to Stay")

        st.write("### 📊 Prediction Confidence")

        st.write(f"**Stay Probability : {stay_probability*100:.2f}%**")

        st.progress(float(stay_probability))

        st.write(f"**Churn Probability : {churn_probability*100:.2f}%**")

        st.progress(float(churn_probability))

        st.divider()

        st.subheader("📌 Actual Result in Dataset")

        if actual_value == "Yes":

            st.error("🔴 Actual Status : Customer Churned")

        else:

            st.success("🟢 Actual Status : Customer Stayed")

st.divider()

st.markdown(
"""
### 💡 About This Project

This application predicts whether a telecom customer is likely to churn using a
Machine Learning model trained on the Telco Customer Churn dataset.

**Technology Stack**

- 🐍 Python
- 📊 Pandas
- 🤖 Scikit-Learn
- 🎨 Streamlit

"""
)

st.markdown(
"""
<hr>

<div style='text-align:center;color:gray'>

Made with ❤️ by <b>Raj</b>

</div>

""",
unsafe_allow_html=True
)