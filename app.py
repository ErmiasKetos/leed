import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import hashlib
import time
from sklearn.cluster import KMeans

# -------------------------------
# Custom CSS for Enterprise Look
# -------------------------------
st.set_page_config(page_title="Enterprise Lead Generation App", layout="wide")
custom_css = """
<style>
/* General Styling */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #f5f5f5;
    color: #333333;
}

/* Header Styling */
h1, h2, h3, h4 {
    color: #333333;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e6e6e6;
}

/* Button Styling */
.stButton button {
    background-color: #007acc;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 0.5em 1em;
}
.stButton button:hover {
    background-color: #005f99;
}

/* DataFrame styling */
.css-1r6slb0 {
    font-size: 0.9em;
}

/* Additional padding for a spacious layout */
.block-container {
    padding: 2rem 1rem;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# -------------------------------
# Helper Functions and Utilities
# -------------------------------

def hash_password(password: str) -> str:
    """Simple hashing function for storing passwords."""
    return hashlib.sha256(password.encode()).hexdigest()

# Dummy user database with hashed passwords
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

def mapping_form(df: pd.DataFrame):
    """
    Presents a form to map required fields to columns in the uploaded file.
    Required fields: 'Company', 'Industry', and 'Engagement' (e.g., Company Size).
    """
    st.subheader("Column Mapping")
    columns = list(df.columns)
    st.write("Detected columns:", columns)
    
    # Provide intelligent defaults if found in the uploaded data.
    default_company = columns.index("Company Name") if "Company Name" in columns else 0
    default_industry = columns.index("Industry") if "Industry" in columns else 0
    default_engagement = columns.index("Approx. Employees") if "Approx. Employees" in columns else 0
    
    mapping = {}
    mapping['Company'] = st.selectbox("Select the column for Company", options=columns, index=default_company)
    mapping['Industry'] = st.selectbox("Select the column for Industry", options=columns, index=default_industry)
    mapping['Engagement'] = st.selectbox("Select the column for Engagement Metric (e.g., Company Size)", options=columns, index=default_engagement)
    
    return mapping

def clean_data_with_mapping(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """
    Cleans and normalizes the CSV data using the user's column mapping.
    Renames columns to canonical names: "Company", "Industry", and "Engagement".
    """
    rename_dict = {
        mapping['Company']: "Company",
        mapping['Industry']: "Industry",
        mapping['Engagement']: "Engagement"
    }
    df = df.rename(columns=rename_dict)
    
    # Drop rows with missing key values.
    df = df.dropna(subset=["Company", "Industry", "Engagement"])
    df["Company"] = df["Company"].astype(str).str.strip()
    df["Industry"] = df["Industry"].astype(str).str.strip()
    
    # Convert Engagement to numeric (remove commas if necessary)
    df["Engagement"] = df["Engagement"].astype(str).str.replace(",", "")
    df["Engagement"] = pd.to_numeric(df["Engagement"], errors="coerce").fillna(0)
    
    return df

@st.cache_data(show_spinner=True)
def enrich_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder for data enrichment.
    In production, integrate with third-party APIs (e.g., Clearbit, LinkedIn).
    """
    with st.spinner("Enriching data with third-party APIs..."):
        time.sleep(2)  # simulate API call delay
    # For demo purposes, add a dummy enrichment column.
    df["EnrichedInfo"] = "Sample Info"
    return df

def calculate_score(row: pd.Series, settings: dict) -> float:
    """
    Calculate lead score using a weighted sum based on multiple parameters.
    
    Parameters used:
    - Engagement: Firmographic measure (e.g., company size)
    - Industry Bonus: Adds bonus points if the lead belongs to a high-priority industry
    
    Formula:
      Score = (Engagement * engagement_weight) + Industry_Bonus
      
    A bonus of 10 points is added if the Industry is in the high-priority list.
    """
    base_score = row.get("Engagement", 0) * settings["engagement_weight"]
    
    # Bonus is applied only if the industry is high priority
    high_priority = settings.get("high_priority_industries", [])
    industry_bonus = 10 if row.get("Industry") in high_priority else 0
    
    total_score = base_score + industry_bonus
    return total_score

def initialize_settings():
    """
    Initialize default settings in session_state if not present, or update missing keys.
    Settings include:
    - engagement_weight: weight for the engagement metric.
    - score_threshold: threshold for high-quality leads.
    - high_priority_industries: list of industries to award bonus points.
    - excluded_industries: list of industries to screen out before analysis.
    """
    defaults = {
        "engagement_weight": 0.01,
        "score_threshold": 50,
        "high_priority_industries": ["Software Development", "Financial Services", "Healthcare"],
        "excluded_industries": []
    }
    if "settings" not in st.session_state:
        st.session_state.settings = defaults
    else:
        for key, value in defaults.items():
            if key not in st.session_state.settings:
                st.session_state.settings[key] = value

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

if st.sidebar.button("Logout"):
    for key in ["authenticated", "username", "role", "data", "mapping", "notes"]:
        st.session_state.pop(key, None)
    st.experimental_rerun()

initialize_settings()

# -------------------------------
# Navigation Sidebar
# -------------------------------
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to", 
    [
        "Data Upload & Mapping", 
        "Dashboard", 
        "Advanced Analytics", 
        "Collaboration Tools", 
        "CRM Integration", 
        "Reporting & Alerts", 
        "Settings"
    ]
)

# -------------------------------
# Page: Data Upload & Mapping
# -------------------------------
if page == "Data Upload & Mapping":
    st.title("Data Upload & Mapping")
    st.write("Upload your LeadFeeder CSV file and map the columns to the required fields for analysis.")
    
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            st.write("### Raw Data Preview")
            st.dataframe(df_raw.head())
            
            mapping = mapping_form(df_raw)
            if st.button("Apply Mapping and Process Data"):
                df_clean = clean_data_with_mapping(df_raw, mapping)
                df_enriched = enrich_data(df_clean)
                df_enriched["Score"] = df_enriched.apply(lambda row: calculate_score(row, st.session_state.settings), axis=1)
                st.session_state.data = df_enriched
                st.session_state.mapping = mapping
                st.success("Data processed successfully!")
                st.write("### Processed Data Preview")
                st.dataframe(df_enriched.head())
        except Exception as e:
            st.error(f"Error processing file: {e}")

# -------------------------------
# Page: Dashboard
# -------------------------------
elif page == "Dashboard":
    st.title("Dashboard")
    if "data" in st.session_state:
        # Apply industry exclusion from settings
        settings = st.session_state.settings
        df = st.session_state.data.copy()
        if settings.get("excluded_industries"):
            df = df[~df["Industry"].isin(settings["excluded_industries"])]
            
        st.subheader("Summary Statistics")
        st.dataframe(df.describe())
        
        st.subheader("Lead Score Distribution")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.hist(df["Score"], bins=20, edgecolor="black")
        ax.set_xlabel("Score")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
        
        st.subheader("Filter by Industry")
        # Create a full list of industries from the (possibly pre-filtered) dataframe
        all_industries = sorted(df["Industry"].dropna().unique())
        search_term = st.text_input("Search industries", "")
        if search_term:
            options = [ind for ind in all_industries if search_term.lower() in ind.lower()]
        else:
            options = all_industries
        # Use only the filtered options for selection
        selected_industries = st.multiselect("Select Industry", options=options, default=options)
        filtered_df = df[df["Industry"].isin(selected_industries)]
        st.write("### Filtered Data Preview")
        st.dataframe(filtered_df.head())
        
        threshold = settings["score_threshold"]
        st.subheader(f"Top Leads (Score >= {threshold})")
        top_leads = df[df["Score"] >= threshold]
        st.dataframe(top_leads)
    else:
        st.warning("No data loaded. Please go to 'Data Upload & Mapping' and upload your CSV file.")

# -------------------------------
# Page: Advanced Analytics
# -------------------------------
elif page == "Advanced Analytics":
    st.title("Advanced Analytics")
    if "data" in st.session_state:
        df = st.session_state.data.copy()
        settings = st.session_state.settings
        if settings.get("excluded_industries"):
            df = df[~df["Industry"].isin(settings["excluded_industries"])]
            
        st.subheader("Predictive Lead Scoring")
        threshold = settings["score_threshold"]
        df["PredictedConversionProbability"] = 1 / (1 + np.exp(- (df["Score"] - threshold) / 10))
        st.write("Sample predictions:")
        st.dataframe(df[["Company", "Score", "PredictedConversionProbability"]].head())
        st.markdown("---")
        
        st.subheader("Segmentation & Clustering")
        features = df[["Engagement", "Score"]]
        try:
            kmeans = KMeans(n_clusters=3, random_state=42).fit(features)
            df["Cluster"] = kmeans.labels_
            st.write("Clustering Results:")
            fig = px.scatter(df, x="Engagement", y="Score", color="Cluster", hover_data=["Company", "Industry"])
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error("Error performing clustering: " + str(e))
        st.markdown("---")
        
        st.subheader("Trend & Time-Series Analysis")
        if "Date" in df.columns:
            try:
                df["Date"] = pd.to_datetime(df["Date"])
                trend_df = df.groupby("Date")["Score"].mean().reset_index()
                fig2 = px.line(trend_df, x="Date", y="Score", title="Average Score Over Time")
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.error("Error in trend analysis: " + str(e))
        else:
            st.info("No date information available for trend analysis.")
    else:
        st.warning("No data loaded. Please upload your CSV file in 'Data Upload & Mapping'.")

# -------------------------------
# Page: Collaboration Tools
# -------------------------------
elif page == "Collaboration Tools":
    st.title("Collaboration Tools")
    if "data" in st.session_state:
        df = st.session_state.data
        if "notes" not in st.session_state:
            st.session_state.notes = {}
        st.write("Select a lead to add or view notes:")
        lead_idx = st.selectbox("Select Lead", options=df.index.tolist(), format_func=lambda idx: df.loc[idx, "Company"])
        note_text = st.text_area("Enter your note for the selected lead:")
        assigned_to = st.text_input("Assign to (e.g., team member name):")
        if st.button("Add Note"):
            note_entry = {"note": note_text, "assigned_to": assigned_to, "timestamp": pd.Timestamp.now()}
            if lead_idx in st.session_state.notes:
                st.session_state.notes[lead_idx].append(note_entry)
            else:
                st.session_state.notes[lead_idx] = [note_entry]
            st.success("Note added!")
        st.markdown("### Existing Notes for Selected Lead")
        if lead_idx in st.session_state.notes:
            for note in st.session_state.notes[lead_idx]:
                st.write(f"**Assigned to:** {note['assigned_to']} | **Time:** {note['timestamp']}")
                st.write(note["note"])
                st.markdown("---")
        else:
            st.info("No notes for this lead yet.")
    else:
        st.warning("No data loaded. Please upload your CSV file in 'Data Upload & Mapping'.")

# -------------------------------
# Page: CRM Integration
# -------------------------------
elif page == "CRM Integration":
    st.title("CRM Integration")
    st.write("This section simulates integration with your CRM system (e.g., Salesforce, HubSpot).")
    if "data" in st.session_state:
        df = st.session_state.data
        st.write("Select leads to push to your CRM:")
        lead_options = df.index.tolist()
        selected_leads = st.multiselect("Select lead indices", options=lead_options, format_func=lambda idx: df.loc[idx, "Company"])
        if st.button("Push Selected Leads to CRM"):
            if selected_leads:
                crm_data = df.loc[selected_leads]
                with st.spinner("Pushing leads to CRM..."):
                    time.sleep(2)
                st.success(f"Successfully pushed {len(crm_data)} leads to the CRM.")
            else:
                st.error("No leads selected. Please select at least one lead.")
    else:
        st.warning("No data loaded. Please upload your CSV file in the 'Data Upload & Mapping' page.")

# -------------------------------
# Page: Reporting & Alerts
# -------------------------------
elif page == "Reporting & Alerts":
    st.title("Reporting & Alerts")
    if "data" in st.session_state:
        df = st.session_state.data
        st.write("Download a report of your processed lead data:")
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(label="Download Lead Report as CSV", data=csv_data, file_name="lead_report.csv", mime="text/csv")
        if (df["Score"] > st.session_state.settings["score_threshold"] * 2).any():
            st.warning("Some leads have exceptionally high scores! Consider following up immediately.")
    else:
        st.warning("No data loaded. Please upload your CSV file in the 'Data Upload & Mapping' page.")

# -------------------------------
# Page: Settings
# -------------------------------
elif page == "Settings":
    st.title("Settings")
    st.write("Adjust the scoring parameters and filtering options. You can also screen out industries from the analysis here.")
    st.markdown("""
    **Scoring Model Explanation:**
    
    The lead score is computed based on multiple parameters:
    
    - **Engagement Metric:**  
      Reflects firmographic data such as company size (e.g., number of employees).  
      *Weight:* `engagement_weight` (default: 0.01)
    
    - **Industry Bonus:**  
      If a lead belongs to a high-priority industry, an extra bonus (e.g., 10 points) is added.
    
    - **Excluded Industries:**  
      Specify industries to screen out entirely from the analysis.
    
    - **Predictive Lead Scoring:**  
      A logistic function simulates the conversion probability based on the overall score.
    
    These parameters can be fine-tuned using historical conversion data.
    """)
    
    engagement_weight = st.number_input("Engagement Weight", value=st.session_state.settings["engagement_weight"], step=0.001, format="%.3f")
    score_threshold = st.number_input("Lead Score Threshold", value=st.session_state.settings["score_threshold"], step=1)
    high_priority_industries = st.text_input("High-Priority Industries (comma separated)", value=", ".join(st.session_state.settings["high_priority_industries"]))
    excluded_industries = st.text_input("Excluded Industries (comma separated)", value=", ".join(st.session_state.settings["excluded_industries"]))
    
    if st.button("Save Settings"):
        st.session_state.settings["engagement_weight"] = engagement_weight
        st.session_state.settings["score_threshold"] = score_threshold
        st.session_state.settings["high_priority_industries"] = [ind.strip() for ind in high_priority_industries.split(",")]
        st.session_state.settings["excluded_industries"] = [ind.strip() for ind in excluded_industries.split(",") if ind.strip()]
        st.success("Settings updated successfully!")
        
    if st.session_state.role == "admin":
        st.subheader("Admin-Only Settings")
        st.info("Additional administrative settings can be placed here.")

# -------------------------------
# End of App - Sidebar Footer
# -------------------------------
st.sidebar.write(f"Logged in as: **{st.session_state.username}** (Role: {st.session_state.role})")
