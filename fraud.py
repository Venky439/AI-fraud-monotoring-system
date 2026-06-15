import streamlit as st
import pandas as pd

st.title("AI Powered Fraud Risk Monitoring System")
st.write("Select a Page from a sidebar")

# Loading the data
uploaded_file = st.file_uploader(
    "Upload Transaction CSV",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(
        uploaded_file
    )

else:
    st.stop()

# bringing the ML model what we saved in notebook
import joblib
model = joblib.load(
    "fraud_model.pkl"
)

# Coverting datetime
df["trans_date_trans_time"] = pd.to_datetime(
    df["trans_date_trans_time"]
)

# Hour
df["hour"] = df["trans_date_trans_time"].dt.hour

# Age Analysis
df["age"] = (
    pd.Timestamp.now().year
    - pd.to_datetime(df["dob"]).dt.year
)

# Distance
df["distance"] = (
    ((df["lat"] - df["merch_lat"]) ** 2 +
     (df["long"] - df["merch_long"]) ** 2) ** 0.5
)

# Dividing the amount into bands
df["amount_band"] = pd.qcut(
    df["amt"],
    q=4,
    labels=[
        "Low",
        "Medium",
        "High",
        "Very High"
    ]
)

# Risk Score
def calculate_risk_score(row):
    score = 0

    if row["amount_band"] == "Very High":
        score += 40

    elif row["amount_band"] == "Medium":
        score += 20

    elif row["amount_band"] == "Low":
        score += 10

    if row["hour"] in [22, 23, 0, 1, 2, 3]:
        score += 30
    return score


df["risk_score"] = df.apply(
    calculate_risk_score,
    axis=1
)

# Risk Level
def get_risk_level(score):

    if score >= 60:
        return "High Risk"

    elif score >= 30:
        return "Medium Risk"

    else:
        return "Low Risk"


df["risk_level"] = df["risk_score"].apply(
    get_risk_level
)
st.write(df.head())

# Risk filter
selected_risk = st.sidebar.selectbox(
    "Select Risk Level",
    ["All"] + list(df["risk_level"].unique())
)

# Category filter
selected_category = st.sidebar.selectbox(
    "Select Category",
    ["All"] + sorted(
        list(df["category"].unique())
    )
)
filtered_df = df.copy()

if selected_risk != "All":
    filtered_df = filtered_df[
        filtered_df["risk_level"] == selected_risk
    ]

if selected_category != "All":
    filtered_df = filtered_df[
        filtered_df["category"] == selected_category
    ]


# State filter
selected_state = st.sidebar.selectbox(
    "Select State",
    ["All"] + sorted(
        list(df["state"].unique())
    )
)

if selected_state != "All":

    filtered_df = filtered_df[
        filtered_df["state"] == selected_state
    ]

# KPI's
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Transactions",
        f"{len(filtered_df):,}"
    )

with col2:
    st.metric(
        "Total Frauds",
        f"{filtered_df['is_fraud'].sum():,}"
    )

with col3:
    fraud_rate = (
        filtered_df["is_fraud"].mean()
    ) * 100

    st.metric(
        "Fraud Rate (%)",
        f"{fraud_rate:.2f}%"
    )

# Fraud Trend Analysis
hourly_trend = (
    filtered_df[
        filtered_df["is_fraud"] == 1
    ]
    .groupby("hour")
    .size()
)
st.subheader(
    "Fraud Trend by Hour"
)
st.line_chart(
    hourly_trend
)

# Category Wise
fraud_category = (
    filtered_df[
        filtered_df["is_fraud"] == 1
    ]
    .groupby("category")
    .size()
    .sort_values(ascending=False)
    .head(10)
)
st.subheader("Top Fraud Categories")
st.bar_chart(fraud_category)  

# Top Fraud Hours
fraud_hours = (
    filtered_df[
        filtered_df["is_fraud"] == 1
    ]
    .groupby("hour")
    .size()
    .sort_values(ascending=False)
)
st.subheader("Fraud Transactions by Hour")
st.bar_chart(fraud_hours)

# Risk Distribution
risk_distribution = (
    filtered_df
    .groupby("risk_level")
    .size()
)
st.subheader("Risk Level Distribution")
st.bar_chart(risk_distribution)

# Transaction Details
st.subheader("Transaction Details")
st.dataframe(
    filtered_df[
        filtered_df["is_fraud"] ==1
        ][
          [
            "amt",
            "category",
            "hour",
            "risk_score",
            "risk_level",
            "is_fraud"
          ]
    ]
)

# Top Fraud States

fraud_states = (
    filtered_df[
        filtered_df["is_fraud"] == 1
    ]
    .groupby("state")
    .size()
    .sort_values(ascending=False)
    .head(10)
)

st.subheader(
    "Top Fraud States"
)

st.bar_chart(
    fraud_states
)

# Fraud Rate by State
state_analysis = (
    filtered_df
    .groupby("state")
    .agg(
        total_transactions=("is_fraud", "count"),
        fraud_transactions=("is_fraud", "sum")
    )
)

state_analysis["fraud_rate"] = (
    state_analysis["fraud_transactions"]
    /
    state_analysis["total_transactions"]
) * 100

top_states = (
    state_analysis
    .sort_values(
        by="fraud_rate",
        ascending=False
    )
    .head(10)
)

st.subheader(
    "Top 10 Fraud Merchants"
)

fraud_merchants = (
    filtered_df[
        filtered_df["is_fraud"] == 1
    ]
    .groupby("merchant")
    .size()
    .sort_values(
        ascending=False
    )
    .head(10)
)

st.bar_chart(
    fraud_merchants
)

st.subheader(
    "Top States by Fraud Rate (%)"
)

st.bar_chart(
    top_states["fraud_rate"]
)


st.header("Fraud Risk Predictor")

amount = st.number_input(
    "Transaction Amount",
    min_value=0.0
)

hour = st.slider(
    "Transaction Hour",
    0,
    23,
    12
)

category = st.selectbox(
    "Transaction Category",
    sorted(
        list(df["category"].unique())
    )
)
if st.button("Predicting Risk"):

    score = 0

    if amount > 500:
        score += 40
    elif amount > 100:
        score += 20
    elif amount > 50:
        score += 10

    if hour in [22, 23, 0, 1, 2, 3]:
        score += 30

    if category == "shopping_net":
        score += 20
    elif category == "misc_net":
        score += 15

    input_data = {
        "amt": amount,
        "hour": hour,
        "risk_score": score,

        "category_food_dining": 0,
        "category_gas_transport": 0,
        "category_grocery_net": 0,
        "category_grocery_pos": 0,
        "category_health_fitness": 0,
        "category_home": 0,
        "category_kids_pets": 0,
        "category_misc_net": 0,
        "category_misc_pos": 0,
        "category_personal_care": 0,
        "category_shopping_net": 0,
        "category_shopping_pos": 0,
        "category_travel": 0
    }

    column_name = f"category_{category}"

    if column_name in input_data:
        input_data[column_name] = 1

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("ML Prediction : Fraud Transaction")
    else:
        st.success("ML Prediction : Genuine Transaction")

# Download Filtered Data
csv = filtered_df.to_csv(index=False)
st.download_button(
    label="Download Filtered Data",
    data=csv,
    file_name="fraud_transactions.csv",
    mime="text/csv"
)

st.header("AI Generated Insights")

highest_fraud_hour = fraud_hours.idxmax()
st.write(
    f"Highest fraud activity is observed at hour {highest_fraud_hour}."
)

highest_category = fraud_category.idxmax()
st.write(
    f"Most fraud-prone category is {highest_category}."
)

highest_state = top_states.index[0]
st.write(
    f"State with highest fraud rate is {highest_state}."
)

