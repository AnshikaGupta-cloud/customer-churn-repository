import streamlit as st
import pandas as pd
import joblib

# -----------------------------------
# LOAD MODEL AND PREPROCESSOR
# -----------------------------------

model = joblib.load("model/logistic_regression.pkl")
preprocessor = joblib.load("model/preprocessor.pkl")


# -----------------------------------
# APP TITLE
# -----------------------------------

st.title("Customer Churn Prediction")

st.write(
    "Enter customer details to predict whether the customer "
    "is likely to churn."
)


# -----------------------------------
# CUSTOMER INPUTS
# -----------------------------------

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

senior_citizen = st.selectbox(
    "Senior Citizen",
    [0, 1]
)

partner = st.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=72,
    value=12
)

phone_service = st.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

multiple_lines = st.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

online_backup = st.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

device_protection = st.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

streaming_tv = st.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=50.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=500.0
)


# -----------------------------------
# PREDICTION
# -----------------------------------

if st.button("Predict Churn"):

    # Create DataFrame from user input
    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })


    # -----------------------------------
    # PREPROCESS INPUT
    # -----------------------------------

    input_processed = preprocessor.transform(input_data)


    # -----------------------------------
    # MAKE PREDICTION
    # -----------------------------------

    prediction = model.predict(input_processed)[0]

    probability = model.predict_proba(
        input_processed
    )[0][1]


    # -----------------------------------
    # DISPLAY PREDICTION
    # -----------------------------------

    if prediction == 1:

        st.error(
            "⚠️ Customer is likely to churn."
        )

    else:

        st.success(
            "✅ Customer is likely to stay."
        )

    st.write(
        f"Churn Probability: {probability:.2%}"
    )


    # -----------------------------------
    # EXPLAINABLE AI
    # -----------------------------------

    st.subheader(
        "🔎 Why did the model make this prediction?"
    )


    # Get feature names after encoding
    feature_names = preprocessor.get_feature_names_out()


    # Get Logistic Regression coefficients
    coefficients = model.coef_[0]


    # Convert processed input to array
    if hasattr(input_processed, "toarray"):
        input_values = input_processed.toarray()[0]
    else:
        input_values = input_processed[0]


    # Calculate contribution
    contributions = input_values * coefficients


    # Create explanation DataFrame
    explanation = pd.DataFrame({
        "Feature": feature_names,
        "Contribution": contributions
    })


    # Sort by strongest influence
    explanation = explanation.sort_values(
        by="Contribution",
        key=abs,
        ascending=False
    )


    # -----------------------------------
    # FACTORS INCREASING CHURN
    # -----------------------------------

    positive_factors = explanation[
        explanation["Contribution"] > 0
    ].head(5)


    # -----------------------------------
    # FACTORS REDUCING CHURN
    # -----------------------------------

    negative_factors = explanation[
        explanation["Contribution"] < 0
    ].head(5)


    # -----------------------------------
    # DISPLAY POSITIVE FACTORS
    # -----------------------------------

    if not positive_factors.empty:

        st.write(
            "🔴 **Factors increasing churn risk:**"
        )

        for _, row in positive_factors.iterrows():

            feature = row["Feature"]

            feature = feature.replace(
                "cat__", ""
            )

            feature = feature.replace(
                "remainder__", ""
            )

            feature = feature.replace(
                "_", " "
            )

            st.write(
                f"• **{feature}** "
                f"(contribution: +{row['Contribution']:.3f})"
            )


    # -----------------------------------
    # DISPLAY NEGATIVE FACTORS
    # -----------------------------------

    if not negative_factors.empty:

        st.write(
            "🟢 **Factors reducing churn risk:**"
        )

        for _, row in negative_factors.iterrows():

            feature = row["Feature"]

            feature = feature.replace(
                "cat__", ""
            )

            feature = feature.replace(
                "remainder__", ""
            )

            feature = feature.replace(
                "_", " "
            )

            st.write(
                f"• **{feature}** "
                f"(contribution: {row['Contribution']:.3f})"
            )
                # -----------------------------------
    # RETENTION RECOMMENDATIONS
    # -----------------------------------

    st.subheader(
        "💡 Recommended Actions to Reduce Churn Risk"
    )

    recommendations = []


    # Contract recommendation
    if contract == "Month-to-month":
        recommendations.append(
            "📄 Offer a discounted one-year or two-year contract "
            "to encourage long-term retention."
        )


    # Monthly charges recommendation
    if monthly_charges > 70:
        recommendations.append(
            "💰 Consider offering a personalized discount or "
            "a more affordable service plan."
        )


    # Tenure recommendation
    if tenure < 12:
        recommendations.append(
            "🎁 Offer an early-customer loyalty benefit or "
            "special introductory offer."
        )


    # Tech support recommendation
    if tech_support == "No":
        recommendations.append(
            "🛠️ Consider offering technical support or a "
            "discounted support package."
        )


    # Online security recommendation
    if online_security == "No":
        recommendations.append(
            "🔐 Offer Online Security as an additional service "
            "to increase customer value."
        )


    # Internet service recommendation
    if internet_service == "Fiber optic":
        recommendations.append(
            "🌐 Check service quality and customer satisfaction "
            "for the fiber-optic connection."
        )


    # Paperless billing recommendation
    if paperless_billing == "Yes":
        recommendations.append(
            "💳 Review the customer's billing experience and "
            "ensure payments are convenient and transparent."
        )


    # If no specific recommendation applies
    if not recommendations:

        st.success(
            "✅ Customer has relatively low-risk characteristics. "
            "Continue providing good service and engagement."
        )

    else:

        for i, recommendation in enumerate(
            recommendations[:5],
            start=1
        ):

            st.write(
                f"**{i}.** {recommendation}"
            )