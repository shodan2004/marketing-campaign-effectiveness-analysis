import pandas as pd
import numpy as np
import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load pre-trained model, features, and scaler
try:
    model = joblib.load('notebooks/best_model.joblib')
    try:
        features = joblib.load('notebooks/model_features.joblib')
    except FileNotFoundError:
        features = ['Age', 'Income', 'Kidhome', 'Teenhome', 'Recency',
                    'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
                    'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
                    'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
                    'NumWebVisitsMonth', 'AcceptedCmp1', 'AcceptedCmp2',
                    'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5',
                    'Days_Since_Enrollment', 'Total_Kids', 'Total_Spent',
                    'Total_Purchases', 'Avg_Spent_per_Purchase', 'Total_Campaign_Accepted',
                    'High_Spender', 'MntWines_Ratio', 'MntFruits_Ratio', 'MntMeatProducts_Ratio',
                    'MntFishProducts_Ratio', 'MntSweetProducts_Ratio', 'MntGoldProds_Ratio',
                    'Education_Basic', 'Education_Graduation', 'Education_Master', 'Education_PhD',
                    'Marital_Status_Alone', 'Marital_Status_Divorced', 'Marital_Status_Married',
                    'Marital_Status_Single', 'Marital_Status_Together', 'Marital_Status_Widow',
                    'Marital_Status_YOLO', 'Family_Status_Alone_1', 'Family_Status_Alone_2',
                    'Family_Status_Divorced_0', 'Family_Status_Divorced_1', 'Family_Status_Divorced_2',
                    'Family_Status_Divorced_3', 'Family_Status_Married_0', 'Family_Status_Married_1',
                    'Family_Status_Married_2', 'Family_Status_Married_3', 'Family_Status_Single_0',
                    'Family_Status_Single_1', 'Family_Status_Single_2', 'Family_Status_Single_3',
                    'Family_Status_Together_0', 'Family_Status_Together_1', 'Family_Status_Together_2',
                    'Family_Status_Together_3', 'Family_Status_Widow_0', 'Family_Status_Widow_1',
                    'Family_Status_Widow_2', 'Family_Status_YOLO_1']
        st.warning("Feature list not found. Using default feature set. Save 'model_features.joblib' after training.")
    try:
        scaler = joblib.load('notebooks/scaler.joblib')
    except FileNotFoundError:
        scaler = None
        st.warning("Scaler not found. Assuming data is unscaled or model handles it.")
except FileNotFoundError:
    st.error("Error: Model 'best_model.joblib' not found. Run training first.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model or files: {str(e)}")
    st.stop()

# Load data for analysis
try:
    df = pd.read_excel("data/processed/encoded/marketing_campaign_ml_ready.xlsx", thousands=',')
except FileNotFoundError:
    st.error("Error: 'marketing_campaign_ml_ready.xlsx' not found. Check path.")
    st.stop()
except Exception as e:
    st.error(f"Error loading Excel file: {str(e)}")
    st.stop()

# Common navigation
def nav_buttons():
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🏠 Home"):
            st.session_state.page = "home"
    with col2:
        if st.button("🔮 Prediction"):
            st.session_state.page = "prediction"
    with col3:
        if st.button("📊 Analysis"):
            st.session_state.page = "analysis"

# Prediction Page
def predict_new_customer():
    st.title("Predict Customer Response")
    with st.expander("Input Customer Details", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=38)
            income = st.number_input("Income", min_value=0, value=150000)
            kidhome = st.number_input("Kidhome", min_value=0, max_value=3, value=0)
            teenhome = st.number_input("Teenhome", min_value=0, max_value=3, value=0)
            recency = st.number_input("Recency", min_value=0, max_value=100, value=0)
        with col2:
            mntwines = st.number_input("MntWines", min_value=0, value=1000)
            mntfruits = st.number_input("MntFruits", min_value=0, value=100)
            mntmeatproducts = st.number_input("MntMeatProducts", min_value=0, value=500)
            mntfishproducts = st.number_input("MntFishProducts", min_value=0, value=200)
            mntsweetproducts = st.number_input("MntSweetProducts", min_value=0, value=100)
        col3, col4 = st.columns(2)
        with col3:
            mntgoldprods = st.number_input("MntGoldProds", min_value=0, value=150)
            numdealspurchases = st.number_input("NumDealsPurchases", min_value=0, value=4)
            numwebpurchases = st.number_input("NumWebPurchases", min_value=0, value=8)
            numcatalogpurchases = st.number_input("NumCatalogPurchases", min_value=0, value=5)
            numstorepurchases = st.number_input("NumStorePurchases", min_value=0, value=6)
        with col4:
            numwebvisitsmonth = st.number_input("NumWebVisitsMonth", min_value=0, value=5)
            acceptedcmp1 = st.selectbox("AcceptedCmp1", [0, 1], index=1)
            acceptedcmp2 = st.selectbox("AcceptedCmp2", [0, 1], index=1)
            acceptedcmp3 = st.selectbox("AcceptedCmp3", [0, 1], index=1)
            acceptedcmp4 = st.selectbox("AcceptedCmp4", [0, 1], index=1)
            acceptedcmp5 = st.selectbox("AcceptedCmp5", [0, 1], index=1)
        customertenuredays = st.number_input("Customer_Tenure_Days", min_value=0, value=700)
        education = st.selectbox("Education", ['Basic', 'Graduation', 'Master', 'PhD'], index=2)
        marital_status = st.selectbox("Marital Status", ['Alone', 'Divorced', 'Married', 'Single', 'Together', 'Widow', 'YOLO'], index=2)

    # --- Encodings ---
    education_encoding = {
        'Basic': {'Education_Basic': 1, 'Education_Graduation': 0, 'Education_Master': 0, 'Education_PhD': 0},
        'Graduation': {'Education_Basic': 0, 'Education_Graduation': 1, 'Education_Master': 0, 'Education_PhD': 0},
        'Master': {'Education_Basic': 0, 'Education_Graduation': 0, 'Education_Master': 1, 'Education_PhD': 0},
        'PhD': {'Education_Basic': 0, 'Education_Graduation': 0, 'Education_Master': 0, 'Education_PhD': 1}
    }
    marital_encoding = {
        'Alone':   {'Marital_Status_Alone': 1, 'Marital_Status_Divorced': 0, 'Marital_Status_Married': 0, 'Marital_Status_Single': 0, 'Marital_Status_Together': 0, 'Marital_Status_Widow': 0, 'Marital_Status_YOLO': 0},
        'Divorced':{'Marital_Status_Alone': 0, 'Marital_Status_Divorced': 1, 'Marital_Status_Married': 0, 'Marital_Status_Single': 0, 'Marital_Status_Together': 0, 'Marital_Status_Widow': 0, 'Marital_Status_YOLO': 0},
        'Married': {'Marital_Status_Alone': 0, 'Marital_Status_Divorced': 0, 'Marital_Status_Married': 1, 'Marital_Status_Single': 0, 'Marital_Status_Together': 0, 'Marital_Status_Widow': 0, 'Marital_Status_YOLO': 0},
        'Single':  {'Marital_Status_Alone': 0, 'Marital_Status_Divorced': 0, 'Marital_Status_Married': 0, 'Marital_Status_Single': 1, 'Marital_Status_Together': 0, 'Marital_Status_Widow': 0, 'Marital_Status_YOLO': 0},
        'Together':{'Marital_Status_Alone': 0, 'Marital_Status_Divorced': 0, 'Marital_Status_Married': 0, 'Marital_Status_Single': 0, 'Marital_Status_Together': 1, 'Marital_Status_Widow': 0, 'Marital_Status_YOLO': 0},
        'Widow':   {'Marital_Status_Alone': 0, 'Marital_Status_Divorced': 0, 'Marital_Status_Married': 0, 'Marital_Status_Single': 0, 'Marital_Status_Together': 0, 'Marital_Status_Widow': 1, 'Marital_Status_YOLO': 0},
        'YOLO':    {'Marital_Status_Alone': 0, 'Marital_Status_Divorced': 0, 'Marital_Status_Married': 0, 'Marital_Status_Single': 0, 'Marital_Status_Together': 0, 'Marital_Status_Widow': 0, 'Marital_Status_YOLO': 1}
    }

    total_kids = kidhome + teenhome
    family_status_encoding = {
        'Alone':    {0: 'Family_Status_Alone_1',     1: 'Family_Status_Alone_2',     2: 'Family_Status_Alone_2',     3: 'Family_Status_Alone_2'},
        'Divorced': {0: 'Family_Status_Divorced_0',  1: 'Family_Status_Divorced_1',  2: 'Family_Status_Divorced_2',  3: 'Family_Status_Divorced_3'},
        'Married':  {0: 'Family_Status_Married_0',   1: 'Family_Status_Married_1',   2: 'Family_Status_Married_2',   3: 'Family_Status_Married_3'},
        'Single':   {0: 'Family_Status_Single_0',    1: 'Family_Status_Single_1',    2: 'Family_Status_Single_2',    3: 'Family_Status_Single_3'},
        'Together': {0: 'Family_Status_Together_0',  1: 'Family_Status_Together_1',  2: 'Family_Status_Together_2',  3: 'Family_Status_Together_3'},
        'Widow':    {0: 'Family_Status_Widow_0',     1: 'Family_Status_Widow_1',     2: 'Family_Status_Widow_2',     3: 'Family_Status_Widow_2'},
        'YOLO':     {0: 'Family_Status_YOLO_1',      1: 'Family_Status_YOLO_1',      2: 'Family_Status_YOLO_1',      3: 'Family_Status_YOLO_1'}
    }
    family_status_key = family_status_encoding[marital_status].get(total_kids, f'Family_Status_{marital_status}_0')
    family_status_cols = {col: 0 for col in [
        'Family_Status_Alone_1','Family_Status_Alone_2',
        'Family_Status_Divorced_0','Family_Status_Divorced_1','Family_Status_Divorced_2','Family_Status_Divorced_3',
        'Family_Status_Married_0','Family_Status_Married_1','Family_Status_Married_2','Family_Status_Married_3',
        'Family_Status_Single_0','Family_Status_Single_1','Family_Status_Single_2','Family_Status_Single_3',
        'Family_Status_Together_0','Family_Status_Together_1','Family_Status_Together_2','Family_Status_Together_3',
        'Family_Status_Widow_0','Family_Status_Widow_1','Family_Status_Widow_2',
        'Family_Status_YOLO_1'
    ]}
    family_status_cols[family_status_key] = 1

    # --- Build new_data ---
    total_spent = mntwines + mntfruits + mntmeatproducts + mntfishproducts + mntsweetproducts + mntgoldprods
    total_purchases = numdealspurchases + numwebpurchases + numcatalogpurchases + numstorepurchases
    avg_spent_per_purchase = (total_spent / total_purchases) if total_purchases > 0 else 0
    total_campaign_accepted = acceptedcmp1 + acceptedcmp2 + acceptedcmp3 + acceptedcmp4 + acceptedcmp5

    def safe_ratio(x): 
        return x / total_spent if total_spent > 0 else 0

    new_data = pd.DataFrame({
        'Age': [age],
        'Income': [income],
        'Kidhome': [kidhome],
        'Teenhome': [teenhome],
        'Recency': [recency],
        'MntWines': [mntwines],
        'MntFruits': [mntfruits],
        'MntMeatProducts': [mntmeatproducts],
        'MntFishProducts': [mntfishproducts],
        'MntSweetProducts': [mntsweetsproducts := mntsweetproducts],  # keep original name downstream
        'MntGoldProds': [mntgoldprods],
        'NumDealsPurchases': [numdealspurchases],
        'NumWebPurchases': [numwebpurchases],
        'NumCatalogPurchases': [numcatalogpurchases],
        'NumStorePurchases': [numstorepurchases],
        'NumWebVisitsMonth': [numwebvisitsmonth],
        'AcceptedCmp1': [acceptedcmp1],
        'AcceptedCmp2': [acceptedcmp2],
        'AcceptedCmp3': [acceptedcmp3],
        'AcceptedCmp4': [acceptedcmp4],
        'AcceptedCmp5': [acceptedcmp5],
        'Days_Since_Enrollment': [customertenuredays],
        'Total_Kids': [total_kids],
        'Total_Spent': [total_spent],
        'Total_Purchases': [total_purchases],
        'Avg_Spent_per_Purchase': [avg_spent_per_purchase],
        'Total_Campaign_Accepted': [total_campaign_accepted],
        'High_Spender': [1 if total_spent > 500 else 0],
        'MntWines_Ratio': [safe_ratio(mntwines)],
        'MntFruits_Ratio': [safe_ratio(mntfruits)],
        'MntMeatProducts_Ratio': [safe_ratio(mntmeatproducts)],
        'MntFishProducts_Ratio': [safe_ratio(mntfishproducts)],
        'MntSweetProducts_Ratio': [safe_ratio(mntsweetsproducts)],
        'MntGoldProds_Ratio': [safe_ratio(mntgoldprods)],
        **education_encoding[education],
        **marital_encoding[marital_status],
        **family_status_cols
    })

    # Ensure all expected features exist (fill missing with 0)
    for feature in features:
        if feature not in new_data.columns:
            new_data[feature] = 0

    # Scale if scaler is provided
    if scaler is not None:
        new_data[features] = scaler.transform(new_data[features])

    threshold = st.slider("Prediction Threshold", 0.1, 0.9, 0.20, 0.05, key="prediction_threshold")
    if st.button("Predict", key="predict_button"):
        try:
            y_prob = model.predict_proba(new_data[features])[:, 1]
            prediction = (y_prob >= threshold).astype(int)
            st.success(f"Predicted Response: {'Yes' if prediction[0] == 1 else 'No'}")
            st.write(f"Probability of Response: {y_prob[0]:.2f}")
            if y_prob[0] < threshold:
                st.warning("Suggestion: Increase spending on top categories or lower the threshold to raise the chance of 'Yes'.")
        except Exception as e:
            st.error(f"Prediction error: {str(e)}. Ensure inputs match model features.")

    # Navigation
    st.markdown("---")
    nav_buttons()

# Analysis Page
def create_dashboard():
    st.title("Campaign Analysis Insights")
    st.sidebar.header("Filters & Settings")
    st.sidebar.slider("Prediction Threshold (display only)", 0.1, 0.9, 0.25, 0.05, key="analysis_threshold")

    education_options = ['Basic', 'Graduation', 'Master', 'PhD']
    marital_options = ['Alone', 'Divorced', 'Married', 'Single', 'Together', 'Widow', 'YOLO']
    selected_education = st.sidebar.multiselect("Education", education_options, default=education_options)
    selected_marital = st.sidebar.multiselect("Marital Status", marital_options, default=marital_options)

    education_filters = [f'Education_{edu}' for edu in selected_education]
    marital_filters = [f'Marital_Status_{mar}' for mar in selected_marital]
    filter_mask = df[education_filters].any(axis=1) & df[marital_filters].any(axis=1)
    filtered_df = df[filter_mask].copy()

    # KPIs (Avg Spending removed)
    col1, col2 = st.columns(2)
    with col1:
        response_rate = filtered_df['Response'].mean() * 100
        st.metric("Response Rate", f"{response_rate:.2f}%")
    with col2:
        total_customers = len(filtered_df)
        st.metric("Total Customers", f"{total_customers:,}")

    # Visualizations
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Response Rate by Education")
        education_response = filtered_df[['Education_Basic', 'Education_Graduation', 'Education_Master', 'Education_PhD']].mean().reset_index()
        education_response.columns = ['Education', 'Response_Rate']
        education_response['Education'] = education_response['Education'].str.replace('Education_', '')
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        sns.barplot(x='Education', y='Response_Rate', data=education_response, ax=ax1, palette='viridis')
        ax1.set_ylabel("Response Rate")
        plt.xticks(rotation=45)
        st.pyplot(fig1)

    with col2:
        st.subheader("Response Rate by Marital Status")
        marital_response = filtered_df[['Marital_Status_Alone', 'Marital_Status_Divorced', 'Marital_Status_Married',
                                       'Marital_Status_Single', 'Marital_Status_Together', 'Marital_Status_Widow',
                                       'Marital_Status_YOLO']].mean().reset_index()
        marital_response.columns = ['Marital_Status', 'Response_Rate']
        marital_response['Marital_Status'] = marital_response['Marital_Status'].str.replace('Marital_Status_', '')
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        sns.barplot(x='Marital_Status', y='Response_Rate', data=marital_response, ax=ax2, palette='magma')
        ax2.set_ylabel("Response Rate")
        plt.xticks(rotation=45)
        st.pyplot(fig2)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Spending Trends by Category")
        spending_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
        spending_data = filtered_df[spending_cols].mean().reset_index()
        spending_data.columns = ['Category', 'Average Spending']
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        sns.barplot(x='Category', y='Average Spending', data=spending_data, ax=ax3, palette='Blues_r')
        ax3.set_ylabel("Average Spending ($)")
        plt.xticks(rotation=45)
        st.pyplot(fig3)

    with col4:
        st.subheader("Campaign Acceptance Heatmap")
        campaign_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
        heatmap_data = filtered_df[campaign_cols].corr()
        fig4, ax4 = plt.subplots(figsize=(5, 4))
        sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', ax=ax4)
        st.pyplot(fig4)

    st.subheader("Feature Importance")
    try:
        feature_importance = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
        fig5, ax5 = plt.subplots(figsize=(8, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), ax=ax5, palette='coolwarm')
        ax5.set_title("Top 10 Feature Importances")
        st.pyplot(fig5)
    except Exception:
        st.info("Model does not expose feature importances.")

    # Navigation
    st.markdown("---")
    nav_buttons()

# Homepage
def home():
    st.title("Marketing Campaign Analysis Dashboard")
    st.markdown("---")
    st.subheader("Welcome to Your Marketing Insights Hub")
    st.write("""
    This dashboard uses a trained ML model to predict customer responses and analyze campaign effectiveness.
    Explore insights across demographics, spending behavior, and campaign responses to make smarter decisions.
    """)
    st.info("💡 Insight: Wine spend tends to correlate strongly with campaign acceptance in many retail datasets.")

    # Navigation
    st.markdown("---")
    nav_buttons()

# Page navigation
if 'page' not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    home()
elif st.session_state.page == "prediction":
    predict_new_customer()
elif st.session_state.page == "analysis":
    create_dashboard()
