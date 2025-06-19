import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="IT Job Analyzer", layout="wide")

st.title("📊 IT Job Market Clustering and Salary Prediction")

# Upload CSV
uploaded_file = st.file_uploader("Upload your IT Job Dataset (CSV)", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.success("File uploaded successfully!")

    # Encode
    le_loc = LabelEncoder()
    le_skill = LabelEncoder()
    le_dom = LabelEncoder()
    df["Location_enc"] = le_loc.fit_transform(df["Location"])
    df["Skills_enc"] = le_skill.fit_transform(df["Skills"])
    df["Domain_enc"] = le_dom.fit_transform(df["Job_Domain"])

    # Cluster
    X_cluster = df[["Experience", "Location_enc", "Skills_enc", "Domain_enc"]]
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_cluster)

    st.subheader("🌀 Cluster Plot")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(data=df, x="Experience", y="Salary_LPA", hue="Cluster", palette="Set2", ax=ax1)
    st.pyplot(fig1)

    # Regression
    X = df[["Experience", "Location_enc", "Skills_enc", "Domain_enc"]]
    y = df["Salary_LPA"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    st.subheader("📈 Random Forest Model Performance")
    st.write(f"Mean Squared Error: {round(mse, 2)}")

    st.subheader("⭐ Feature Importance")
    importances = rf.feature_importances_
    fig2, ax2 = plt.subplots()
    sns.barplot(x=importances, y=X.columns, ax=ax2)
    st.pyplot(fig2)

    st.subheader("🎯 Predict Salary for a New Candidate")
    exp = st.slider("Experience (Years)", 0, 10, 3)
    loc = st.selectbox("Location", df["Location"].unique())
    skill = st.selectbox("Skill", df["Skills"].unique())
    domain = st.selectbox("Job Domain", df["Job_Domain"].unique())

    sample = pd.DataFrame({
        "Experience": [exp],
        "Location_enc": [le_loc.transform([loc])[0]],
        "Skills_enc": [le_skill.transform([skill])[0]],
        "Domain_enc": [le_dom.transform([domain])[0]]
    })

    pred_salary = rf.predict(sample)[0]
    st.success(f"💼 Estimated Salary: ₹ {round(pred_salary, 2)} LPA")
