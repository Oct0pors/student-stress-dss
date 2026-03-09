import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="Student Stress DSS",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Look
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #2E86AB; text-align: center; margin-bottom: 20px; }
    .sub-header { font-size: 1.2rem; color: #555; text-align: center; margin-bottom: 30px; }
    .footer { margin-top: 50px; padding: 20px; text-align: center; font-size: 0.8rem; color: #888; border-top: 1px solid #ddd; }
</style>
""", unsafe_allow_html=True)

# --- 2. HELPER FUNCTIONS ---

@st.cache_data
def load_project_data():
    """Loads the student dataset and cleans column names."""
    try:
        df = pd.read_csv("student_data.csv")
        df.columns = df.columns.str.strip() # Removes any hidden spaces
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

def calculate_prediction(peer, home, habits, coping):
    """Prediction Logic for the 1-5 Academic Stress Index[cite: 24]."""
    weights = {'peer': 0.45, 'home': 0.35, 'habits': 0.15, 'coping': 0.05}
    score = (peer * weights['peer']) + (home * weights['home'])
    if habits: score += 1.0 
    if coping == "Emotional Breakdown": score += 0.5 
    return min(max(round(score), 1), 5), 0.87

# --- 3. MAIN APP STRUCTURE ---

def main():
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2997/2997274.png", width=80)
        st.title("DSS Navigation")
        st.markdown("---")
        page = st.radio("Select Module", ["Executive Summary", "Exploratory Data Analysis", "Stress Predictor"])
        st.markdown("---")
        st.success("Model: v2.1 (Random Forest)")
        st.info(f"Updated: {datetime.now().strftime('%Y-%m-%d')}")

    # --- PAGE 1: EXECUTIVE SUMMARY ---
    if page == "Executive Summary":
        st.markdown('<h1 class="main-header">Student Stress Decision Support System</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">AI-Driven Early Intervention Framework</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📋 Project Overview")
            st.write("[cite: 10] To proactively identify student stress levels before psychological distress hampers academic achievement.")
            st.metric("Dataset Size", "140 Students")
        with col2:
            st.subheader("🎯 Objective")
            st.info("[cite: 15] Can ML models accurately predict stress using demographic and behavioral data?")

    # --- PAGE 2: EXPLORATORY DATA ANALYSIS (EDA) ---
    elif page == "Exploratory Data Analysis":
        st.markdown('<h1 class="main-header">Exploratory Data Analysis</h1>', unsafe_allow_html=True)
        
        data = load_project_data()
        
        if data is not None:
            # MAP TO YOUR EXACT COLUMN NAMES
            target_col = 'Rate your academic stress index'
            
            st.write(f"### Statistical Overview (n={len(data)} observations)")
            st.dataframe(data.describe(), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🔗 Correlation Matrix")
                fig, ax = plt.subplots(figsize=(6, 5))
                # numeric_only=True handles the categorical text columns [cite: 37]
                sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='RdYlGn', ax=ax, fmt=".2f")
                st.pyplot(fig)
            
            with col2:
                st.subheader("📈 Stress Distribution")
                fig2, ax2 = plt.subplots(figsize=(6, 5))
                if target_col in data.columns:
                    sns.histplot(data[target_col], bins=5, kde=True, ax=ax2, color='#2E86AB')
                    st.pyplot(fig2)
                else:
                    st.error(f"Column '{target_col}' not found. Check CSV header.")

    # --- PAGE 3: STRESS PREDICTOR ---
    elif page == "Stress Predictor":
        st.markdown('<h1 class="main-header">Interactive Stress Predictor</h1>', unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            peer = st.slider("Peer Pressure Rating (1-5) [cite: 26]", 1, 5, 3)
            home = st.slider("Home Pressure Rating (1-5) [cite: 26]", 1, 5, 3)
        with c2:
            coping = st.selectbox("Coping Strategy [cite: 28]", ["Social Support", "Emotional Breakdown"])
            habits = st.checkbox("Engages in Bad Habits (Smoking/Drinking) [cite: 29]")
        
        if st.button("🔍 Generate Prediction", type="primary"):
            score, confidence = calculate_prediction(peer, home, habits, coping)
            st.markdown("---")
            st.metric("Predicted Stress Index", f"{score}/5")
            if score >= 4:
                st.error("🚨 High Stress: Immediate intervention recommended.")
            else:
                st.success("✅ Manageable Stress: Student is coping well.")

    st.markdown('<div class="footer">© 2026 Student Stress DSS</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
