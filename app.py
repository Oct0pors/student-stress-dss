import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(page_title="Student Stress DSS", layout="wide")

# Sidebar Navigation
st.sidebar.title("DSS Navigation")
page = st.sidebar.radio("Go to:", ["Executive Summary", "Exploratory Data Analysis", "Stress Predictor"])

# --- PAGE 1: EXECUTIVE SUMMARY ---
if page == "Executive Summary":
    st.title("Project Proposal: Predicting Student Stress Level")
    st.write("**Date:** March 9, 2026")
    
    st.header("Background")
    st.info("""
    Recent studies estimate that 10% to 30% of students worldwide suffer from 
    stress-related impairments (WHO, 2024). This study proposes an AI-driven 
    classification model to enable early intervention.
    """)
    
    st.subheader("Problem Statement")
    st.write("Can machine learning models accurately predict students' academic stress levels using demographic, academic, and behavioral data?")

# --- PAGE 2: EXPLORATORY DATA ANALYSIS (EDA) ---
elif page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    st.write("Visualizing patterns and relationships among student variables.")

    # Simulated data based on your 140 observations
    data = pd.DataFrame({
        'Academic Stress Index': np.random.randint(1, 6, 140),
        'Peer Pressure': np.random.randint(1, 6, 140),
        'Home Pressure': np.random.randint(1, 6, 140),
        'Study Hours': np.random.randint(1, 12, 140)
    })

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        st.write("Pearson correlation used to determine significant predictors.")

    with col2:
        st.subheader("Stress Level Distribution")
        fig2, ax2 = plt.subplots()
        sns.histplot(data['Academic Stress Index'], bins=5, kde=True, ax=ax2)
        st.pyplot(fig2)

# --- PAGE 3: STRESS PREDICTOR ---
elif page == "Stress Predictor":
    st.title("Interactive Stress Prediction Tool")
    st.write("Enter student details to calculate the predicted Academic Stress Index (1-5).")

    with st.expander("Input Student Data", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            stage = st.selectbox("Academic Stage", ["Undergraduate", "High School"])
            peer = st.slider("Peer Pressure Rating", 1, 5, 3)
            home = st.slider("Home Pressure Rating", 1, 5, 3)
        with c2:
            env = st.selectbox("Study Environment", ["Peaceful", "Noisy", "Disrupted"])
            coping = st.selectbox("Coping Strategy", ["Social Support", "Emotional Breakdown"])
            habits = st.checkbox("Engages in Bad Habits (Smoking/Drinking)")

    # Logic for Recommendation (Decision Support)
    if st.button("Generate Prediction"):
        # Placeholder for your classification model logic
        base_stress = (peer + home) / 2
        if habits: base_stress += 1
        if coping == "Emotional Breakdown": base_stress += 1
        
        final_score = min(int(base_stress), 5)
        
        st.subheader(f"Predicted Stress Index: {final_score}")
        
        if final_score >= 4:
            st.error("Result: High Stress Level. Early intervention is highly recommended.")
        elif final_score == 3:
            st.warning("Result: Moderate Stress Level. Monitor student workload.")
        else:
            st.success("Result: Low Stress Level. Continue current support systems.")
