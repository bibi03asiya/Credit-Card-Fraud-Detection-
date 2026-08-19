import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, classification_report, precision_score, recall_score, f1_score

from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------
# Page Config
# ------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Custom CSS for Premium Dark Look
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background: radial-gradient(circle at top right, #1e2130, #0e1117);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #161922;
        border-right: 1px solid #2d323e;
    }
    
    /* Precise Metric Cards */
    div[data-testid="stMetric"] {
        background: rgba(30, 33, 48, 0.7);
        backdrop-filter: blur(10px);
        padding: 15px 20px !important;
        border-radius: 12px;
        border: 1px solid rgba(0, 212, 255, 0.1);
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.3);
        transition: all 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 212, 255, 0.4);
    }
    
    /* Heading Styles */
    h1 {
        font-size: 2.8rem !important;
        letter-spacing: -1px;
    }
    h1, h2, h3 {
        background: linear-gradient(90deg, #00d4ff, #0080ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }
    
    /* Compact Layout */
    .block-container {
        padding-top: 3rem !important;
        padding-bottom: 3rem !important;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(45deg, #00d4ff 0%, #0080ff 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 0.8rem;
        letter-spacing: 1.2px;
        transition: all 0.2s ease;
    }
    
    /* Precision Dataframes */
    .stDataFrame {
        font-size: 0.85rem;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("💳 Credit Card Fraud Detection System")


# ------------------------
# Create Realistic Dataset
# ------------------------

np.random.seed(42)

data = {
"Amount":np.random.randint(100,50000,800),
"Time":np.random.randint(1,1000,800),
"V1":np.random.randn(800),
"V2":np.random.randn(800),
"V3":np.random.randn(800),
"V4":np.random.randn(800),
"Fraud":np.random.choice([0,1],800,p=[0.93,0.07])
}

df = pd.DataFrame(data)


# ------------------------
# Train Model
# ------------------------

X = df.drop("Fraud",axis=1)
y = df["Fraud"]

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

X_train,X_test,y_train,y_test = train_test_split(
X_res,y_res,test_size=0.2,random_state=42
)

log_model = LogisticRegression(max_iter=1000)
rf_model = RandomForestClassifier()

log_model.fit(X_train,y_train)
rf_model.fit(X_train,y_train)


# Accuracy and Metrics
log_pred = log_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

metrics_data = {
    "Logistic Regression": {
        "Accuracy": accuracy_score(y_test, log_pred),
        "Precision": precision_score(y_test, log_pred),
        "Recall": recall_score(y_test, log_pred),
        "F1-Score": f1_score(y_test, log_pred)
    },
    "Random Forest": {
        "Accuracy": accuracy_score(y_test, rf_pred),
        "Precision": precision_score(y_test, rf_pred),
        "Recall": recall_score(y_test, rf_pred),
        "F1-Score": f1_score(y_test, rf_pred)
    }
}

log_acc = metrics_data["Logistic Regression"]["Accuracy"]
rf_acc = metrics_data["Random Forest"]["Accuracy"]


# ------------------------
# Sidebar
# ------------------------

st.sidebar.title("🛡️ SecureGuard AI")
st.sidebar.write("Next-gen Fraud Analytics")

# Boxed Security Quote
st.sidebar.markdown("""
<div style="
    background: rgba(0, 212, 255, 0.1);
    border: 1px solid #00d4ff;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 20px;
">
    <h3 style="margin: 0; color: #00d4ff;">🛡️ stay safe</h3>
    <p style="margin: 5px 0 0 0; font-size: 0.8rem; font-weight: bold; letter-spacing: 1px; color: #ffffff;">
        BE ALERT. BE SECURE
    </p>
</div>
""", unsafe_allow_html=True)

menu = st.sidebar.selectbox(
"Navigation",
["Dashboard","Data Explorer","Detect Fraud","Model Comparison","Analytics"]
)

st.sidebar.divider()
st.sidebar.subheader("Project Information")
st.sidebar.info("""
**Credit Card Fraud Detection**
- **Models**: RF & Logistic Regression
- **Data**: Synthetic (SMOTE Balanced)
- **Engine**: Scikit-Learn
""")
st.sidebar.caption("© 2026 Antigravity Systems")


# ------------------------
# Dashboard
# ------------------------

if menu == "Dashboard":

    st.header("📊 Dashboard Overview")

    fraud = df["Fraud"].value_counts()

    col1,col2,col3,col4 = st.columns(4)

    with col1:
        st.metric("Total Transactions",len(df))

    with col2:
        st.metric("Fraud Cases",fraud.get(1, 0))

    with col3:
        st.metric("Legitimate",fraud.get(0, 0))

    with col4:
        st.metric("Fraud Rate",
        f"{round((fraud.get(1, 0)/len(df))*100,2)}%")


    st.subheader("Model Performance & Fraud Distribution")

    # Layout with 3 columns for balanced visualization
    col_dash1, col_dash2, col_dash3 = st.columns(3)

    with col_dash1:
        st.write("<p style='text-align: center; color: #00d4ff; font-weight: bold;'>Logistic Regression</p>", unsafe_allow_html=True)
        fig_log, ax_log = plt.subplots(figsize=(3, 3), facecolor='#0e1117')
        ax_log.set_facecolor('#0e1117')
        ax_log.pie([log_acc, 1-log_acc], colors=['#0080ff', '#2d323e'], 
                   startangle=90, counterclock=False)
        # Centered percentage text
        ax_log.text(0, 0, f"{log_acc*100:.0f}%", ha='center', va='center', 
                    color='white', fontsize=20, fontweight='bold')
        centre_circle = plt.Circle((0,0),0.75,fc='#0e1117')
        fig_log.gca().add_artist(centre_circle)
        ax_log.axis('equal')
        st.pyplot(fig_log)

    with col_dash2:
        st.write("<p style='text-align: center; color: #00d4ff; font-weight: bold;'>Random Forest</p>", unsafe_allow_html=True)
        fig_rf, ax_rf = plt.subplots(figsize=(3, 3), facecolor='#0e1117')
        ax_rf.set_facecolor('#0e1117')
        ax_rf.pie([rf_acc, 1-rf_acc], colors=['#00d4ff', '#2d323e'], 
                  startangle=90, counterclock=False)
        # Centered percentage text
        ax_rf.text(0, 0, f"{rf_acc*100:.0f}%", ha='center', va='center', 
                   color='white', fontsize=20, fontweight='bold')
        centre_circle = plt.Circle((0,0),0.75,fc='#0e1117')
        fig_rf.gca().add_artist(centre_circle)
        ax_rf.axis('equal')
        st.pyplot(fig_rf)

    with col_dash3:
        st.write("<p style='text-align: center; color: #ff4b4b; font-weight: bold;'>Fraud Rate Analysis</p>", unsafe_allow_html=True)
        fig_fraud, ax_fraud = plt.subplots(figsize=(3, 3), facecolor='#0e1117')
        ax_fraud.set_facecolor('#0e1117')
        ax_fraud.pie(
            [fraud.get(0, 0), fraud.get(1, 0)],
            labels=["Legitimate", "Fraud"],
            autopct="%1.1f%%",
            textprops={'color':"w", 'fontsize': 10},
            colors=['#0080ff', '#ff4b4b'],
            startangle=140,
            explode=(0, 0.1)  # Explode the fraud slice
        )
        ax_fraud.axis('equal')
        st.pyplot(fig_fraud)

# ------------------------
# Data Explorer
# ------------------------

elif menu == "Data Explorer":
    st.header("🔍 Data Explorer")
    st.write("Preview of the generated transaction data:")
    st.dataframe(df.head(20), use_container_width=True)
    
    st.subheader("Dataset Statistics")
    st.write(df.describe())


# ------------------------
# Detect Fraud
# ------------------------

elif menu == "Detect Fraud":

    st.header("💳 Detect Fraud Transaction")
    st.write("Enter transaction details below for real-time risk assessment.")

    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            amount = st.number_input("Amount (₹)", 100, step=500)
            v1 = st.number_input("V1", 0.0, format="%.2f")
        with c2:
            time = st.number_input("Time Index", 1)
            v2 = st.number_input("V2", 0.0, format="%.2f")
        with c3:
            v3 = st.number_input("V3", 0.0, format="%.2f")
            v4 = st.number_input("V4", 0.0, format="%.2f")


    if st.button("Check Transaction"):

        input_data = np.array([[amount,time,v1,v2,v3,v4]])

        prediction = rf_model.predict(input_data)
        prob = rf_model.predict_proba(input_data)[0][1]

        # Force suspicious condition
        if amount > 40000:
            prediction = [1]
            prob = 0.99

        st.divider()
        
        if prediction[0] == 1:
            st.error(f"🚨 Fraud Transaction Detected (Probability: {prob*100:.1f}%)")
            st.warning("⚠️ Action Required: This transaction exceeds safety thresholds and shows suspicious patterns.")
        else:
            st.success(f"✅ Legitimate Transaction (Fraud Probability: {prob*100:.1f}%)")
            st.info("ℹ️ Transaction matches standard user behavior patterns.")
# ------------------------
# Model Comparison
# ------------------------

elif menu == "Model Comparison":
    st.header("📊 Model Performance Comparison")
    st.write("In-depth evaluation of Random Forest vs. Logistic Regression models.")

    # Convert metrics to DataFrame for plotting
    comp_df = pd.DataFrame(metrics_data).T
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Performance Metrics")
        st.dataframe(comp_df.style.background_gradient(cmap="Blues"), use_container_width=True)
    
    with col2:
        st.subheader("Accuracy Comparison")
        fig, ax = plt.subplots(facecolor='#0e1117')
        ax.set_facecolor('#0e1117')
        comp_df["Accuracy"].plot(kind='bar', color=['#ff4b4b', '#00d4ff'], ax=ax)
        ax.tick_params(colors='white')
        plt.xticks(rotation=0)
        st.pyplot(fig)

    st.divider()
    
    st.subheader("Detailed Metric Comparison")
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0e1117')
    ax.set_facecolor('#0e1117')
    comp_df.drop("Accuracy", axis=1).plot(kind='bar', ax=ax, color=['#00d4ff', '#0080ff', '#4b00ff'])
    ax.tick_params(colors='white')
    plt.xticks(rotation=0)
    plt.legend(loc="lower right")
    st.pyplot(fig)


# ------------------------
# Analytics
# ------------------------

elif menu == "Analytics":

    st.header("📈 Advanced Analytics")
    st.write("In-depth evaluation of model performance and feature significance.")

    # Row 1: Confusion Matrix and ROC Curve
    col_ana1, col_ana2 = st.columns(2)

    with col_ana1:
        st.subheader("Confusion Matrix")
        pred = rf_model.predict(X_test)
        cm = confusion_matrix(y_test,pred)

        fig,ax = plt.subplots(facecolor='#0e1117')
        ax.set_facecolor('#0e1117')
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.tick_params(colors='white')
        ax.set_xlabel("Predicted", color="white")
        ax.set_ylabel("Actual", color="white")
        st.pyplot(fig)

    with col_ana2:
        st.subheader("Model Comparison (ROC Curve)")
        fig_roc, ax_roc = plt.subplots(facecolor='#0e1117')
        ax_roc.set_facecolor('#0e1117')
        
        # Random Forest ROC
        rf_probs = rf_model.predict_proba(X_test)[:, 1]
        fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
        roc_auc_rf = auc(fpr_rf, tpr_rf)
        
        # Logistic Regression ROC
        log_probs = log_model.predict_proba(X_test)[:, 1]
        fpr_log, tpr_log, _ = roc_curve(y_test, log_probs)
        roc_auc_log = auc(fpr_log, tpr_log)
        
        plt.plot(fpr_rf, tpr_rf, color='#00d4ff', lw=2, label=f'RF (AUC = {roc_auc_rf:.2f})')
        plt.plot(fpr_log, tpr_log, color='#ff4b4b', lw=2, label=f'LogReg (AUC = {roc_auc_log:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        
        ax_roc.tick_params(colors='white')
        ax_roc.xaxis.label.set_color('white')
        ax_roc.yaxis.label.set_color('white')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        st.pyplot(fig_roc)

    st.divider()

    # Row 2: Feature Importance
    st.subheader("Feature Importance (Random Forest)")
    
    importance = rf_model.feature_importances_
    features = X.columns
    importance_df = pd.DataFrame({"Feature": features, "Importance": importance}).sort_values(by="Importance", ascending=False)

    fig_feat, ax_feat = plt.subplots(figsize=(10, 4), facecolor='#0e1117')
    ax_feat.set_facecolor('#0e1117')
    sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis", ax=ax_feat)
    ax_feat.tick_params(colors='white')
    ax_feat.xaxis.label.set_color('white')
    ax_feat.yaxis.label.set_color('white')
    st.pyplot(fig_feat)
