import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import time

# -------------------------------
# Helper Functions and Utilities
# -------------------------------

def hash_password(password: str) -> str:
    """Simple hashing function for storing passwords."""
    return hashlib.sha256(password.encode()).hexdigest()

# Dummy user database with hashed passwords
# In production, integrate with an enterprise-grade authentication provider.
users = {
    "admin": hash_password("admin123"),
    "sales": hash_password("sales123")
}

user_roles = {
    "admin": "admin",
    "sales": "sales"
}

def authenticate(username: str, password: str) -> bool:
    """Check username and password."""
    if username in users and hash_password(password) == users[username]:
        return True
    return False

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic data cleaning and normalization.
    Expects at least: 'Company', 'Industry', 'Visits', 'TimeSpent'
    """
    required_cols = ["Company", "Industry", "Visits", "TimeSpent"]
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Missing required column: {col}")
            st.stop()
    # Drop rows with missing key values
    df = df.dropna(subset=required_cols)
    df["Company"] = df["Company"].astype(str).str.strip()
    df["Industry"] = df["Industry"].astype(str).str.strip()
    # Ensure numeric columns are numbers
    df["Visits"] = pd.to_numeric(df["Visits"], errors="coerce").fillna(0)
    df["TimeSpent"] = pd.to_numeric(df["TimeSpent"], errors="coerce").fillna(0)
    return df

@st.cache_data(show_spinner=True)
def enrich_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder for data enrichment.
    In production, integrate with third‑party APIs (e.g., Clearbit, LinkedIn, ZoomInfo).
    """
    with st.spinner("Enriching data with third‑party APIs..."):
        time.sleep(2)  # simulate API call delay
    # For demo purposes, add a dummy column.
    df["EnrichedInfo"] = "Sample Info"
    return df

def calculate_score(row: pd.Series, settings: dict) -> float:
    """
    Calculate lead score using a simple weighted sum.
    Assumes the row has 'Visits' and 'TimeSpent' columns.
    """
    score = row.get("Visits", 0) * settings["visit_weight"] + row.get("TimeSpent", 0) * settings["time_weight"]
    return score

def initialize_settings():
    """Initialize default settings in session_state if not present."""
    if "settings" not in st.session_state:
        st.session_state.settings = {
            "visit_weight": 1.0,
            "time_weight": 0.1,
            "score_threshold": 50
        }

# -------------------------------
# User Authentication Section
# -------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.sidebar.header("Login")
    username_input = st.sidebar.text_input("Username")
    password_input = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if authenticate(username_input, password_input):
            st.session_state.authenticated = True
            st.session_state.username = username_input
            st.session_state.role = user_roles.get(username_input, "sales")
            st.success(f"Welcome, {username_input}!")
        else:
            st.error("Incorrect username or password. Please try again.")
    st.stop()

# Add a logout button in the sidebar
if st.sidebar.button("Logout"):
    for key in ["authenticated", "username", "role", "data"]:
        st.session_state.pop(key, None)
    st.experimental_rerun()

# Initialize settings if needed
initialize_settings()

# -------------------------------
# Navigation Sidebar
# -------------------------------
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", 
                        ["Dashboard", "Data Upload & Processing", "CRM Integration", "Reporting & Alerts", "Settings"])

# -------------------------------
# Page: Data Upload & Processing
# -------------------------------
if page == "Data Upload & Processing":
    st.title("Data Upload & Processing")
    st.write("Upload your LeadFeeder CSV file. The data must include at least the following columns: 'Company', 'Industry', 'Visits', and 'TimeSpent'.")
    
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### Raw Data Preview", df.head())
            df = clean_data(df)
            df = enrich_data(df)
            # Calculate score for each lead based on current settings
            df["Score"] = df.apply(lambda row: calculate_score(row, st.session_state.settings), axis=1)
            st.session_state.data = df  # Save processed data in session state
            st.success("Data processed successfully!")
            
            # Display interactive filtering options
            st.subheader("Filter Your Data")
            industries = df["Industry"].unique()
            selected_industries = st.multiselect("Select Industry", options=list(industries), default=list(industries))
            filtered_df = df[df["Industry"].isin(selected_industries)]
            st.write("### Filtered Data Preview", filtered_df.head())
            
        except Exception as e:
            st.error(f"Error processing file: {e}")

# -------------------------------
# Page: Dashboard
# -------------------------------
elif page == "Dashboard":
    st.title("Dashboard")
    if "data" in st.session_state:
        df = st.session_state.data
        st.subheader("Summary Statistics")
        st.write(df.describe())
        
        # Histogram of Lead Scores
        st.subheader("Lead Score Distribution")
        fig, ax = plt.subplots()
        ax.hist(df["Score"], bins=20, edgecolor="black")
        ax.set_xlabel("Score")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
        
        # Display top leads based on the score threshold set in settings
        threshold = st.session_state.settings["score_threshold"]
        st.subheader(f"Top Leads (Score >= {threshold})")
        top_leads = df[df["Score"] >= threshold]
        st.dataframe(top_leads)
    else:
        st.warning("No data loaded. Please go to the 'Data Upload & Processing' page and upload your CSV.")

# -------------------------------
# Page: CRM Integration
# -------------------------------
elif page == "CRM Integration":
    st.title("CRM Integration")
    st.write("This section would integrate with your CRM system (e.g., Salesforce, HubSpot).")
    if "data" in st.session_state:
        df = st.session_state.data
        st.write("Select leads to push to your CRM:")
        # Let the user select leads by company name.
        lead_options = df.index.tolist()
        selected_leads = st.multiselect("Select lead indices", options=lead_options, 
                                          format_func=lambda idx: df.loc[idx, "Company"])
        if st.button("Push Selected Leads to CRM"):
            if selected_leads:
                # Here you would implement the API call to your CRM system.
                crm_data = df.loc[selected_leads]
                # Simulate CRM integration delay
                with st.spinner("Pushing leads to CRM..."):
                    time.sleep(2)
                st.success(f"Successfully pushed {len(crm_data)} leads to the CRM.")
            else:
                st.error("No leads selected. Please select at least one lead.")
    else:
        st.warning("No data loaded. Please upload your CSV first.")

# -------------------------------
# Page: Reporting & Alerts
# -------------------------------
elif page == "Reporting & Alerts":
    st.title("Reporting & Alerts")
    if "data" in st.session_state:
        df = st.session_state.data
        st.write("Generate a downloadable report of your lead data.")
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Lead Report as CSV",
            data=csv_data,
            file_name="lead_report.csv",
            mime="text/csv"
        )
        # Example: Check for exceptionally high scores and show an alert.
        if (df["Score"] > st.session_state.settings["score_threshold"] * 2).any():
            st.warning("Some leads have exceptionally high scores! Consider following up immediately.")
    else:
        st.warning("No data loaded. Please upload your CSV first.")

# -------------------------------
# Page: Settings
# -------------------------------
elif page == "Settings":
    st.title("Settings")
    st.write("Adjust the lead scoring parameters and other configurations.")
    
    # Display current settings and allow modifications
    visit_weight = st.number_input("Visit Weight", value=st.session_state.settings["visit_weight"], step=0.1)
    time_weight = st.number_input("Time Spent Weight", value=st.session_state.settings["time_weight"], step=0.1)
    score_threshold = st.number_input("Lead Score Threshold", value=st.session_state.settings["score_threshold"], step=1)
    
    if st.button("Save Settings"):
        st.session_state.settings["visit_weight"] = visit_weight
        st.session_state.settings["time_weight"] = time_weight
        st.session_state.settings["score_threshold"] = score_threshold
        st.success("Settings updated successfully!")
        
    # Optionally, if the user is an admin, allow further configuration
    if st.session_state.role == "admin":
        st.subheader("Admin-Only Settings")
        st.info("Additional administrative settings can be placed here.")

# -------------------------------
# End of App
# -------------------------------
st.sidebar.write(f"Logged in as: **{st.session_state.username}** (Role: {st.session_state.role})")
