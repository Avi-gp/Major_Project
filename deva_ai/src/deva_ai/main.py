import sqlite_patch
import json
import pandas as pd
import streamlit as st
import os
from crew import DevaAi
from utils import save_uploadedfile,Efile, extract_data_with_regex, extract_preprocessing_info,preprocess_for_json, parse_json_safely, extract_value  ,extract_feature_engineering_info             
from streamlit_components.data_ingestion_output import display_data_ingestion_results
from streamlit_components.data_preprocessing_output import display_preprocessing_results
from streamlit_components.feature_engineering_output import display_feature_engineering_results
from streamlit_components.insight_generation_output import display_insight_generation_results

# Set the page configuration as the first Streamlit command
st.set_page_config(page_title="DEVA AI : AI-Powered Data Analysis", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "Home"

# Add session state variables for tracking process status
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False
    
if "crew_executed" not in st.session_state:
    st.session_state.crew_executed = False
    
if "target_column_selected" not in st.session_state:
    st.session_state.target_column_selected = False

#-----------------------------------------------------------------------------------------------#
def main():
    # Custom CSS styling remains the same
    st.markdown("""
        <style>
            /* DEVA AI - Professional Enhanced CSS Styling */

            /* Import modern fonts */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@400;500;600;700&display=swap');

            /* Professional color scheme and variables */
            :root {
                --primary-blue: #3b82f6;
                --secondary-blue: #60a5fa;
                --accent-blue: #93c5fd;
                --dark-bg: #0f1419;
                --card-bg: #1f2937;
                --text-primary: #f9fafb;
                --text-secondary: #d1d5db;
                --text-muted: #9ca3af;
                --border-color: #374151;
                --success-color: #10b981;
                --warning-color: #f59e0b;
                --error-color: #ef4444;
                --glass-bg: rgba(31, 41, 55, 0.8);
            }

            /* Main app background with modern gradient */
            .stApp {
                background: linear-gradient(135deg, #0f1419 0%, #1a202c 30%, #2d3748 70%, #374151 100%) !important;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
            }

            /* Professional title styling - clean white text */
            .title {
                font-family: 'Inter', 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
                font-weight: 700;
                text-align: center;
                font-size: 2.5rem;
                background: linear-gradient(135deg, 
                    #ffffff 0%, 
                    #f1f5f9 30%, 
                    #e2e8f0 70%, 
                    #cbd5e1 100%);
                -webkit-background-clip: text;
                background-clip: text;
                padding: 30px 0px 20px 0px;
                letter-spacing: -0.5px;
                position: relative;
                transition: all 0.6s cubic-bezier(0.4, 0, 0.2, 1);
                cursor: default;
                margin: 0;
            }

            .title:hover {
                background: linear-gradient(135deg, 
                    #fbbf24 0%, 
                    #f59e0b 25%, 
                    #d97706 40%, 
                    #60a5fa 60%, 
                    #3b82f6 80%, 
                    #1e40af 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                transform: scale(1.01) translateY(-1px);
                filter: drop-shadow(0 0 25px rgba(251, 191, 36, 0.3)) drop-shadow(0 0 25px rgba(59, 130, 246, 0.3));
                }


            /* Professional subheader styling - white with hover transition */
            .subheader {
                font-family: 'Inter', sans-serif;
                font-weight: 600;
                font-size: 1.5rem;
                color: #ffffff;
                padding: 20px 0px 15px 0px;
                margin-bottom: 25px;
                letter-spacing: -0.3px;
                position: relative;
                transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
                cursor: default;
            }

            .subheader:hover {
                background: linear-gradient(90deg, 
                    #059669 0%, 
                    #10b981 25%, 
                    #0ea5e9 60%, 
                    #0284c7 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                transform: translateX(7px) scale(1.005);
                filter: drop-shadow(0 0 16px rgba(16, 185, 129, 0.4));
            }

            .subheader::before {
                content: '';
                position: absolute;
                bottom: 5px;
                left: 0;
                width: 40px;
                height: 2px;
                background: #64748b;
                border-radius: 2px;
                transition: all 0.5s ease;
            }

            .subheader:hover::before {
                width: 110px;
                background: linear-gradient(90deg, #059669, #10b981, #0ea5e9);
                box-shadow: 0 0 12px rgba(16, 185, 129, 0.5);
            }

            /* Glass morphism card design */
            .card {
                background: linear-gradient(145deg, 
                    rgba(31, 41, 55, 0.9) 0%, 
                    rgba(37, 47, 63, 0.8) 50%, 
                    rgba(45, 55, 72, 0.7) 100%);
                backdrop-filter: blur(20px) saturate(180%);
                border-radius: 16px;
                padding: 28px;
                box-shadow: 
                    0 15px 35px rgba(0, 0, 0, 0.3),
                    0 0 0 1px rgba(59, 130, 246, 0.1),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
                margin-bottom: 28px;
                border: 1px solid rgba(59, 130, 246, 0.15);
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                position: relative;
                overflow: hidden;
            }

            .card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 1px;
                background: linear-gradient(90deg, 
                    transparent 0%, 
                    rgba(59, 130, 246, 0.4) 50%, 
                    transparent 100%);
            }

            .card:hover {
                transform: translateY(-5px) scale(1.01);
                box-shadow: 
                    0 20px 40px rgba(0, 0, 0, 0.4),
                    0 0 0 1px rgba(59, 130, 246, 0.2),
                    inset 0 1px 0 rgba(255, 255, 255, 0.15);
                border-color: rgba(59, 130, 246, 0.3);
            }

            /* Team card - professional clean design */
            .team-card-simple {
                background: linear-gradient(145deg, 
                    rgba(31, 41, 55, 0.95) 0%, 
                    rgba(37, 47, 63, 0.9) 100%);
                backdrop-filter: blur(15px) saturate(150%);
                border-radius: 20px;
                padding: 32px 24px;
                text-align: center;
                margin: 16px 12px;
                box-shadow: 
                    0 10px 25px rgba(0, 0, 0, 0.25),
                    0 0 0 1px rgba(59, 130, 246, 0.1),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(59, 130, 246, 0.12);
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                position: relative;
                overflow: hidden;
                min-height: 200px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
            }

            .team-card-simple::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 2px;
                background: linear-gradient(90deg, 
                    var(--primary-blue) 0%, 
                    var(--secondary-blue) 50%, 
                    var(--accent-blue) 100%);
                opacity: 0;
                transition: opacity 0.4s ease;
            }

            .team-card-simple:hover::before {
                opacity: 1;
            }

            .team-card-simple:hover {
                transform: translateY(-8px) scale(1.03);
                box-shadow: 
                    0 18px 35px rgba(0, 0, 0, 0.3),
                    0 0 0 1px rgba(59, 130, 246, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.2);
                border-color: rgba(59, 130, 246, 0.25);
            }

            .team-avatar {
                display: flex;
                justify-content: center;
                margin-bottom: 18px;
            }

            .avatar-placeholder {
                width: 75px;
                height: 75px;
                border-radius: 50%;
                background: linear-gradient(135deg, 
                    var(--primary-blue) 0%, 
                    var(--secondary-blue) 50%, 
                    var(--accent-blue) 100%);
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 22px;
                font-weight: 700;
                color: white;
                box-shadow: 
                    0 8px 20px rgba(59, 130, 246, 0.25),
                    inset 0 2px 4px rgba(255, 255, 255, 0.2);
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                position: relative;
                overflow: hidden;
            }

            .team-card-simple:hover .avatar-placeholder {
                transform: scale(1.15) rotate(3deg);
                box-shadow: 
                    0 12px 25px rgba(59, 130, 246, 0.35),
                    inset 0 2px 4px rgba(255, 255, 255, 0.25);
            }

            .team-name {
                font-family: 'Inter', sans-serif;
                font-size: 1.25rem; /* Professional size */
                font-weight: 600;
                color: #ffffff !important;
                margin: 16px 0 10px 0;
                letter-spacing: -0.2px;
                text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
                transition: color 0.3s ease;
            }

            .team-card-simple:hover .team-name {
                color: var(--secondary-blue) !important;
            }

            .team-role {
                font-size: 0.875rem; /* Professional size */
                font-weight: 500;
                color: var(--secondary-blue);
                margin-bottom: 0;
                text-transform: uppercase;
                letter-spacing: 0.8px;
                opacity: 0.9;
            }
                
            
            /* GitHub link styling for team cards */
            .github-link {
                transition: all 0.3s ease !important;
                cursor: pointer !important;
            }

            .github-link:hover {
                color: var(--secondary-blue) !important;
                transform: scale(1.05) !important;
            }

            .team-card-simple a:hover .avatar-placeholder {
                background: linear-gradient(135deg, 
                    #10b981 0%, 
                    var(--primary-blue) 50%, 
                    var(--secondary-blue) 100%) !important;
                box-shadow: 
                    0 8px 20px rgba(16, 185, 129, 0.3),
                    0 0 15px rgba(59, 130, 246, 0.2) !important;
            }

            .team-card-simple a:hover .team-name {
                color: var(--secondary-blue) !important;
                text-shadow: 0 0 8px rgba(96, 165, 250, 0.4) !important;
            }

            /* Enhanced text styling */
            .description-text {
                color: var(--text-secondary) !important;
                font-size: 1rem; /* Professional size */
                line-height: 1.6;
                font-weight: 400;
                font-family: 'Inter', sans-serif;
                text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
            }

            .highlight-text {
                color: #ffffff !important;
                font-weight: 600;
                background: linear-gradient(90deg, 
                    rgba(59, 130, 246, 0.12) 0%, 
                    rgba(59, 130, 246, 0.04) 100%);
                padding: 14px 18px;
                border-radius: 10px;
                border-left: 3px solid var(--primary-blue);
                margin: 20px 0;
                font-family: 'Inter', sans-serif;
                box-shadow: 
                    0 3px 10px rgba(59, 130, 246, 0.1),
                    inset 0 1px 0 rgba(255, 255, 255, 0.08);
                font-size: 1.1rem; /* Professional size */
                letter-spacing: -0.1px;
                transition: all 0.3s ease;
            }

            .highlight-text:hover {
                color: var(--secondary-blue) !important;
                transform: translateX(3px);
            }

            /* Fixed width navigation buttons */
            .stButton>button {
                background: transparent !important;
                color: #ffffff !important;
                font-family: 'Inter', sans-serif !important;
                border-radius: 12px !important;
                padding: 12px 24px !important;
                border: 1.5px solid rgba(255, 255, 255, 0.2) !important;
                font-weight: 500 !important;
                font-size: 0.95rem !important;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
                letter-spacing: 0.2px !important;
                position: relative !important;
                overflow: hidden !important;
                width: 200px !important; /* Fixed width */
                max-width: 200px !important; /* Maximum width */
                text-align: left !important;
                white-space: nowrap !important; /* Prevent text wrapping */
            }

            .stButton>button::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, 
                    transparent 0%, 
                    rgba(59, 130, 246, 0.1) 50%, 
                    transparent 100%);
                transition: left 0.5s ease;
            }

            .stButton>button:hover::before {
                left: 100%;
            }

            .stButton>button:hover {
                background: linear-gradient(135deg, 
                    rgba(59, 130, 246, 0.15) 0%, 
                    rgba(96, 165, 250, 0.1) 100%) !important;
                color: var(--secondary-blue) !important;
                border-color: rgba(59, 130, 246, 0.4) !important;
                transform: translateX(5px) !important;
                box-shadow: 
                    0 5px 15px rgba(59, 130, 246, 0.2),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
            }

            /* Analyze button special styling */
            .analyze-button {
                background: linear-gradient(135deg, 
                    #10b981 0%, 
                    #059669 50%, 
                    #047857 100%) !important;
                color: white !important;
                padding: 16px 32px !important;
                font-size: 1.1rem !important;
                font-weight: 600 !important;
                border-radius: 12px !important;
                margin: 25px 0 !important;
                border: none !important;
                box-shadow: 
                    0 8px 20px rgba(16, 185, 129, 0.25),
                    inset 0 2px 4px rgba(255, 255, 255, 0.15) !important;
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
                display: block !important;
                width: 100% !important;
                text-align: center !important;
                position: relative !important;
                overflow: hidden !important;
                text-transform: uppercase !important;
                letter-spacing: 1px !important;
            }

            .analyze-button:hover {
                background: linear-gradient(135deg, 
                    #059669 0%, 
                    #047857 50%, 
                    #065f46 100%) !important;
                box-shadow: 
                    0 12px 25px rgba(16, 185, 129, 0.35),
                    inset 0 2px 4px rgba(255, 255, 255, 0.2) !important;
                transform: translateY(-3px) scale(1.02) !important;
                letter-spacing: 1.5px !important;
            }

            /* Sidebar modern styling */
            .css-1d391kg {
                background: linear-gradient(180deg, 
                    rgba(15, 20, 25, 0.95) 0%, 
                    rgba(31, 41, 55, 0.9) 100%) !important;
                backdrop-filter: blur(15px) !important;
                border-right: 1px solid rgba(59, 130, 246, 0.15) !important;
            }

            /* Enhanced metric cards */
            .metric-card {
                background: linear-gradient(145deg, 
                    rgba(45, 45, 45, 0.9) 0%, 
                    rgba(37, 37, 37, 0.8) 100%);
                backdrop-filter: blur(10px);
                border-radius: 12px;
                padding: 20px;
                box-shadow: 
                    0 6px 15px rgba(0, 0, 0, 0.25),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
                text-align: center;
                margin: 10px 6px;
                border: 1px solid rgba(59, 130, 246, 0.12);
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            }

            .metric-card:hover {
                transform: translateY(-5px) scale(1.03);
                box-shadow: 
                    0 12px 25px rgba(0, 0, 0, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.15);
                border-color: rgba(59, 130, 246, 0.2);
            }

            .metric-value {
                font-size: 1.75rem; /* Professional size */
                font-weight: 700;
                color: var(--secondary-blue);
                text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
                margin-bottom: 10px;
                font-family: 'Inter', sans-serif;
            }

            .metric-label {
                font-size: 0.8rem; /* Professional size */
                color: var(--text-muted);
                margin-top: 6px;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.8px;
            }

            /* Footer modern styling */
            .footer {
                text-align: center;
                color: var(--text-muted);
                padding: 35px 20px;
                border-radius: 12px;
                font-size: 0.875rem; /* Professional size */
                border-top: 1px solid rgba(59, 130, 246, 0.15);
                margin-top: 45px;
                background: linear-gradient(135deg, 
                    rgba(15, 20, 25, 0.8) 0%, 
                    rgba(31, 41, 55, 0.6) 100%);
                backdrop-filter: blur(10px);
                font-family: 'Inter', sans-serif;
                font-weight: 400;
            }

            .footer a {
                color: var(--secondary-blue);
                text-decoration: none;
                transition: all 0.3s ease;
                font-weight: 500;
            }

            .footer a:hover {
                color: var(--accent-blue);
                text-shadow: 0 0 6px rgba(147, 197, 253, 0.4);
            }

            /* Enhanced expander styling */
            .streamlit-expanderHeader {
                background: linear-gradient(135deg, 
                    rgba(31, 41, 55, 0.9) 0%, 
                    rgba(37, 47, 63, 0.8) 100%) !important;
                backdrop-filter: blur(10px) !important;
                border-radius: 12px !important;
                color: #ffffff !important;
                padding: 16px 20px !important;
                font-weight: 600 !important;
                border: 1px solid rgba(59, 130, 246, 0.15) !important;
                transition: all 0.4s ease !important;
                font-family: 'Inter', sans-serif !important;
                font-size: 1rem !important;
                box-shadow: 0 3px 10px rgba(0, 0, 0, 0.15) !important;
            }

            .streamlit-expanderHeader:hover {
                background: linear-gradient(135deg, 
                    rgba(37, 47, 63, 0.9) 0%, 
                    rgba(45, 55, 72, 0.8) 100%) !important;
                color: var(--secondary-blue) !important;
                border-color: rgba(59, 130, 246, 0.25) !important;
                transform: translateY(-1px) !important;
                box-shadow: 0 6px 15px rgba(0, 0, 0, 0.2) !important;
            }

            /* Target section styling */
            .target-section {
                background: linear-gradient(145deg, 
                    rgba(42, 42, 42, 0.9) 0%, 
                    rgba(30, 30, 30, 0.8) 100%);
                backdrop-filter: blur(15px);
                border-radius: 16px;
                padding: 25px;
                margin: 20px 0;
                border: 1px solid rgba(59, 130, 246, 0.15);
                box-shadow: 
                    0 8px 20px rgba(0, 0, 0, 0.25),
                    inset 0 1px 0 rgba(255, 255, 255, 0.08);
            }

            .target-info {
                color: var(--secondary-blue);
                font-size: 1rem; /* Professional size */
                margin-bottom: 3px
                line-height: 1.6;
                font-weight: 400;
            }

            /* File upload styling */
            .stFileUploader {
                border: 2px dashed rgba(59, 130, 246, 0.25) !important;
                border-radius: 12px !important;
                padding: 20px !important;
                background: rgba(31, 41, 55, 0.4) !important;
                backdrop-filter: blur(10px) !important;
                transition: all 0.3s ease !important;
            }

            .stFileUploader:hover {
                border-color: rgba(59, 130, 246, 0.4) !important;
                background: rgba(31, 41, 55, 0.6) !important;
            }

            /* Progress bar styling */
            .stProgress > div > div {
                background: linear-gradient(90deg, 
                    var(--primary-blue) 0%, 
                    var(--secondary-blue) 100%) !important;
                border-radius: 6px !important;
                box-shadow: 0 2px 6px rgba(59, 130, 246, 0.25) !important;
            }

            /* Spinner enhancement */
            .stSpinner > div {
                border-top-color: var(--primary-blue) !important;
                border-right-color: var(--secondary-blue) !important;
            }

            /* Success/Error message styling */
            .stSuccess {
                background: rgba(16, 185, 129, 0.08) !important;
                border: 1px solid rgba(16, 185, 129, 0.25) !important;
                border-radius: 10px !important;
                backdrop-filter: blur(10px) !important;
            }

            .stError {
                background: rgba(239, 68, 68, 0.08) !important;
                border: 1px solid rgba(239, 68, 68, 0.25) !important;
                border-radius: 10px !important;
                backdrop-filter: blur(10px) !important;
            }

            .stWarning {
                background: rgba(245, 158, 11, 0.08) !important;
                border: 1px solid rgba(245, 158, 11, 0.25) !important;
                border-radius: 10px !important;
                backdrop-filter: blur(10px) !important;
            }

            /* Benefit list styling */
            .benefit-item {
                margin: 6px 0;
                padding: 6px 0;
                border-bottom: 1px solid rgba(59, 130, 246, 0.08);
                color: var(--text-secondary);
                font-size: 0.95rem; /* Professional size */
                line-height: 1.5;
            }

            .benefit-item:last-child {
                border-bottom: none;
            }

            /* Download button enhancement */
            .stDownloadButton > button {
                background: rgba(77, 139, 240, 0.9) !important;
                color: white !important;
                padding: 14px 28px !important;
                font-weight: 600 !important;
                border-radius: 10px !important;
                border: none !important;
                margin: 20px auto !important;
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
                box-shadow: 0 6px 12px rgba(77, 139, 240, 0.25) !important;
                text-transform: uppercase !important;
                letter-spacing: 0.3px !important;
                font-size: 0.9rem !important;
            }

            .stDownloadButton > button:hover {
                background: rgba(58, 120, 221, 0.95) !important;
                box-shadow: 0 8px 18px rgba(77, 139, 240, 0.35) !important;
                transform: translateY(-2px) scale(1.03) !important;
            }

            /* Responsive design */
            @media (max-width: 768px) {
                .title {
                    font-size: 2rem;
                    padding: 20px 0px 15px 0px;
                }
                
                .subheader {
                    font-size: 1.25rem;
                    padding: 16px 0px 12px 0px;
                }
                
                .card {
                    padding: 20px;
                    margin-bottom: 20px;
                }
                
                .team-card-simple {
                    margin: 12px 6px;
                    padding: 28px 18px;
                    min-height: 180px;
                }
                
                .avatar-placeholder {
                    width: 65px;
                    height: 65px;
                    font-size: 20px;
                }
                
                .team-name {
                    font-size: 1.1rem;
                }
                
                .team-role {
                    font-size: 0.8rem;
                }
            }

            /* Scrollbar styling */
            ::-webkit-scrollbar {
                width: 6px;
            }

            ::-webkit-scrollbar-track {
                background: rgba(31, 41, 55, 0.4);
                border-radius: 3px;
            }

            ::-webkit-scrollbar-thumb {
                background: linear-gradient(180deg, 
                    var(--primary-blue) 0%, 
                    var(--secondary-blue) 100%);
                border-radius: 3px;
            }

            ::-webkit-scrollbar-thumb:hover {
                background: linear-gradient(180deg, 
                    var(--secondary-blue) 0%, 
                    var(--accent-blue) 100%);
            }

            /* Animation keyframes */
            @keyframes fadeInUp {
                from {
                    opacity: 0;
                    transform: translateY(15px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            @keyframes slideInRight {
                from {
                    opacity: 0;
                    transform: translateX(15px);
                }
                to {
                    opacity: 1;
                    transform: translateX(0);
                }
            }

            /* Apply animations */
            .card {
                animation: fadeInUp 0.5s ease-out;
            }

            .team-card-simple {
                animation: slideInRight 0.5s ease-out;
            }

            /* Focus states for accessibility */
            .stButton>button:focus {
                outline: 2px solid var(--secondary-blue);
                outline-offset: 2px;
            }

            .stSelectbox > div > div:focus {
                border-color: var(--primary-blue) !important;
                box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.15) !important;
            }
                
            
        </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; background: transparent; border-radius: 12px; margin-bottom: 10px;">
            <div style="display: inline-flex; align-items: center; gap: 10px;">
                <svg width="24" height="24" viewBox="0 0 24 24" style="filter: drop-shadow(0 0 4px rgba(96, 165, 250, 0.4));">
                    <polygon points="12,2 22,8 22,16 12,22 2,16 2,8" fill="none" stroke="#60A5FA" stroke-width="1"/>
                    <circle cx="12" cy="12" r="6" fill="none" stroke="#60A5FA" stroke-width="1.5"/>
                    <circle cx="12" cy="12" r="3" fill="#60A5FA"/>
                    <circle cx="12" cy="12" r="1" fill="#1F2937"/>
                </svg>
                <span style="background: linear-gradient(135deg, #F8FAFC 0%, #E2E8F0 50%, #CBD5E1 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-size: 32px; font-weight: 700; font-family: 'Roboto', 'Arial', sans-serif; letter-spacing:1.5px;">DEVA AI</span>
            </div>
            <div style="margin-top: 5px;">
                <span style="background: linear-gradient(90deg, #374151, #6B7280); padding: 3px 12px; border-radius: 15px; color: #E5E7EB; font-size: 11px; font-weight: 500; font-family: 'Poppins', sans-serif; letter-spacing: 1px; box-shadow: 0 2px 6px rgba(96, 165, 250, 0.2); border: 1px solid #4B5563;">✨ UNVEILING INSIGHTS ✨</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### Navigation")
        if st.button("🏠 Home"):
            st.session_state.page = "Home"
        if st.button("ℹ️ About"):
            st.session_state.page = "About"
        if st.button("❓ Help"):
            st.session_state.page = "Help"
        
        st.markdown("---")
        st.markdown("### Contact")
        st.markdown("📧 support@devaai.com")
        st.markdown("🌐 www.devaai.com")

    page = st.session_state.page

    # Main content based on selected page
    if page == "Home":
        # Title with icon
        st.markdown('<h1 class="title">🧑‍💼 DEVA AI : AI-Powered Data Analysis </h1>', unsafe_allow_html=True)
        
        # Introduction
        col1, col2 = st.columns([1.3,1])
        with col1:
            st.markdown("""
                <div class="card">
                    <h3 class="highlight-text">Unlock the Power of Your Data</h3>
                    <p class="description-text">Upload your dataset and let our advanced AI transform raw numbers into clear, actionable insights. No coding required – just upload and discover what your data has been trying to tell you.</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="card">
                    <h4 class="highlight-text">Key Benefits</h4>
                    <ul>
                        <li class="benefit-item"><strong>Instant Exploration</strong> – Understand your data in seconds</li>
                        <li class="benefit-item"><strong>Smart Visualizations</strong> – See patterns you might have missed</li>
                        <li class="benefit-item"><strong>Actionable Insights</strong> – Make data-driven decisions with confidence</li>   
                    </ul>
                </div>
            """, unsafe_allow_html=True)

        # Upload file section            
        st.markdown('<h2 class="subheader">Upload Your Dataset</h2>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

        if uploaded_file:
            try:
                # Save file and get absolute path
                file_path = save_uploadedfile(uploaded_file)
                
                if file_path:
                    st.success(f"File successfully saved at: {file_path}")
                    st.session_state.file_uploaded = True
                    st.session_state['file_path'] = file_path
                    st.session_state['uploaded_file_name'] = uploaded_file.name
                    
                    # Check file extension
                    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                    if file_extension not in ['.csv', '.xlsx']:
                        st.error("Unsupported file format. Please upload CSV or Excel file.")
                        st.stop()
                    
                    # Add analyze button after file is uploaded
                    if st.button("✨ Unleash AI Analysis", key="analyze_button", help="Let our AI reveal insights hidden in your data"):
                        # This button click will trigger the CrewAI execution
                        if not st.session_state.get('crew_executed', False):
                            # Create DevaAi instance
                            deva = DevaAi()
                            inputs = {
                                "file_path": file_path,
                                "file_name": uploaded_file.name,
                                "preprocessed_file_path": "",
                                "preprocessed_file_name": "",
                                "engineered_file_path": "",
                                "engineered_file_name": "",
                            }

                            with st.spinner('Processing dataset... Please wait'):
                                crew_output = deva.crew().kickoff(inputs=inputs)
                                st.session_state['crew_results'] = crew_output
                                st.session_state['last_file'] = file_path
                                st.session_state.crew_executed = True
                                
                            st.rerun()
                    
                    # Display CrewAI results if already executed
                    if st.session_state.get('crew_executed', False) and 'crew_results' in st.session_state:
                        crew_output = st.session_state['crew_results']
                        
                        # Display CrewAI results
                        if hasattr(crew_output, 'tasks_output'):
                            progress_bar = st.progress(0)
                            
                            if len(crew_output.tasks_output) > 0:
                                st.success("✅ Data successfully loaded and analyzed!")
                                display_data_ingestion_results(crew_output)
                                progress_bar.progress(50)
                            
                            if len(crew_output.tasks_output) > 1:
                                display_preprocessing_results(crew_output.tasks_output[1])
                                progress_bar.progress(75)
                                st.success("✅ Data preprocessing completed!")

                            if len(crew_output.tasks_output) > 2:
                                display_feature_engineering_results(crew_output.tasks_output[2])
                                progress_bar.progress(100)
                                st.success("✅ Feature Engineering completed!")

                                # Extract the raw text from feature engineering output
                                feature_eng_info = crew_output.tasks_output[2].raw
                                # Get engineered file info using Efile function
                                engineered_file_path, engineered_file_name = Efile(feature_eng_info)

                                if not engineered_file_path or not engineered_file_name:
                                    st.error("Failed to extract engineered file information")
                                    st.stop()

                                # Store in session state for later use
                                st.session_state['engineered_file_path'] = engineered_file_path
                                st.session_state['engineered_file_name'] = engineered_file_name
                            progress_bar.empty()
                            
                            # Target Column Selection (moved after CrewAI output)
                            st.markdown('<h2 class="subheader">Select Target Variable for Insights</h2>', unsafe_allow_html=True)
                            st.markdown("""
                                <div class="target-section">
                                    <p class="target-info">Select the column you want to analyze as your target variable.</p>
                            """, unsafe_allow_html=True)
                            
                            # Load engineered dataset for column selection using the file name
                            engineered_file_path = st.session_state.get('engineered_file_path')
                            engineered_file_name = st.session_state.get('engineered_file_name')
                            
                            engineered_df = pd.read_csv(engineered_file_path) if engineered_file_name.endswith('.csv') else pd.read_excel(engineered_file_path)
                            columns = engineered_df.columns.tolist()
                            
                            # Add a placeholder option as first item
                            columns_with_placeholder = ["--Select a target column--"] + columns
                                
                            # Create a unique key for the selectbox to ensure it refreshes properly
                            select_key = f"target_column_select_{engineered_file_name}"
                            
                            # Use selectbox with placeholder as first option
                            target_column = st.selectbox(
                                "Select the target column:",
                                options=columns_with_placeholder,
                                index=0,  # Default to the placeholder
                                key=select_key
                            )
                            
                            # Check if user has selected a valid column (not the placeholder)
                            if target_column != "--Select a target column--":
                                st.session_state['target_column'] = target_column
                                st.session_state.target_column_selected = True
                                
                                # Only show insights when a target is selected
                                if st.session_state.target_column_selected:
                                    # Show progress container
                                    insight_progress = st.empty()
                                    insight_status = st.empty()
                                    
                                    with st.spinner('🔍 Generating comprehensive data insights...'):
                                        # Initialize progress
                                        insight_progress.progress(0)
                                        insight_status.info('Starting insight generation...')
                                        
                                        try:
                                            # Update progress at 25%
                                            insight_progress.progress(25)
                                            insight_status.info('Loading and preparing data...')
                                            
                                            # Update progress at 50%
                                            insight_progress.progress(50)
                                            insight_status.info('Analyzing relationships and patterns...')
                                            
                                            # Update progress at 75%
                                            insight_progress.progress(75)
                                            insight_status.info('Generating visualizations...')
                                            
                                            # Pass the extracted engineered file path and selected target column
                                            display_insight_generation_results(engineered_file_path, target_column)
                                            
                                            # Complete progress
                                            insight_progress.progress(100)
                                            insight_status.success('✨ Insights generated successfully!')
                                            
                                            # Clear progress indicators after completion
                                            insight_progress.empty()
                                            
                                        except Exception as e:
                                            insight_status.error(f"Error during insight generation: {str(e)}")
                                            st.exception(e)
                            else:
                                # Show a message prompting user to select a target column
                                st.info("Please select a target column to generate insights")
                                    
                            st.markdown("""</div>""", unsafe_allow_html=True)
                                
                                
                        else:
                            st.warning("Task output not structured as expected. Displaying raw results:")
                            try:
                                raw_content = crew_output.raw.strip("`")
                                raw_content = preprocess_for_json(raw_content)
                                result = json.loads(raw_content)
                                st.json(result)
                            except Exception as parse_error:
                                st.error(f"Error parsing raw output: {str(parse_error)}")
                                st.write(crew_output.raw)

                else:
                    st.error("Failed to save the uploaded file.")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)
        

    elif page == "About":
        st.markdown('<h1 class="title">About DEVA AI</h1>', unsafe_allow_html=True)
        st.markdown("""
            <div class="card">
                <h3 class="highlight-text">Our Mission</h3>
                <p class="description-text">DEVA AI is on a mission to democratize data analysis and make it accessible to everyone, 
                regardless of their technical background. We believe that the insights hidden in data should be available to all, 
                not just those with specialized training.</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="card">
                <h3 class="highlight-text">Our Technology</h3>
                <p class="description-text">Powered by cutting-edge AI models and the innovative CrewAI framework, DEVA AI transforms 
                complex data analysis into an intuitive experience. Our platform bridges the gap between raw data and meaningful insights, 
                eliminating the steep learning curve traditionally associated with data science.</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="card">
                <h3 class="highlight-text">Meet Our Team</h3>
                <p class="description-text">Behind DEVA AI is a diverse team of data scientists, AI researchers, and software engineers 
                united by a shared vision: making data analysis intuitive, powerful, and accessible.</p>
            </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("""
                <div class="team-card-simple">
                    <a href="https://github.com/priyankaangane" target="_blank" style="text-decoration: none; color: inherit;">
                        <div class="team-avatar">
                            <div class="avatar-placeholder github-link">PA</div>
                        </div>
                        <h4 class="team-name github-link">Priyanka Angane</h4>
                    </a>
                    <p class="team-role">Team Leader & Developer</p>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
                <div class="team-card-simple">
                    <a href="https://github.com/Avi-gp" target="_blank" style="text-decoration: none; color: inherit;">
                        <div class="team-avatar">
                            <div class="avatar-placeholder github-link">SG</div>
                        </div>
                        <h4 class="team-name github-link">Suryansh Gupta</h4>
                    </a>
                    <p class="team-role">Developer</p>
                </div>
            """, unsafe_allow_html=True)

        # Second row
        col3, col4 = st.columns(2, gap="large")

        with col3:
            st.markdown("""
                <div class="team-card-simple">
                    <a href="https://github.com/Veritate1311" target="_blank" style="text-decoration: none; color: inherit;">
                        <div class="team-avatar">
                            <div class="avatar-placeholder github-link">VS</div>
                        </div>
                        <h4 class="team-name github-link">Vasudha Singh</h4>
                    </a>
                    <p class="team-role">Developer</p>
                </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown("""
                <div class="team-card-simple">
                    <a href="https://github.com/rxshoumoun" target="_blank" style="text-decoration: none; color: inherit;">
                        <div class="team-avatar">
                            <div class="avatar-placeholder github-link">RJ</div>
                        </div>
                        <h4 class="team-name github-link">Rohan Jadhav</h4>
                    </a>
                    <p class="team-role">Developer</p>
                </div>
            """, unsafe_allow_html=True)

        
    elif page == "Help":
        st.markdown('<h1 class="title">Help & Documentation</h1>', unsafe_allow_html=True)
    
        with st.expander("How to Use DEVA AI"):
            st.markdown("""
                <div class="description-text">
                <ol style="line-height: 2; padding-left: 20px;">
                    <li><span style="color: #60a5fa; font-weight: 600;">Upload your data</span>: Click on the file upload button on the home page and select a CSV or Excel file from your computer.</li>
                    <li><span style="color: #60a5fa; font-weight: 600;">Click Unleash AI Analysis</span>: After your file is uploaded, click the "Unleash AI Analysis" button to start the AI analysis process.</li>
                    <li><span style="color: #60a5fa; font-weight: 600;">Processing</span>: Our AI engine will automatically analyze your data, identifying patterns and preparing visualizations.</li>
                    <li><span style="color: #60a5fa; font-weight: 600;">Select Target Variable</span>: Choose the column you want to focus on for deeper insights.</li>
                    <li><span style="color: #60a5fa; font-weight: 600;">Explore insights</span>: Navigate through the generated sections to discover different perspectives on your data.</li>
                    <li><span style="color: #60a5fa; font-weight: 600;">Download results</span>: Save any visualizations or insights for your presentations, reports, or further analysis.</li>
                </ol>
                </div>
            """, unsafe_allow_html=True)
        
        with st.expander("Supported File Formats"):
            st.markdown("""
                <div class="description-text">
                <p><strong class="highlight-text">DEVA AI currently supports:</strong></p>
                <ul style="line-height: 2; padding-left: 20px;">
                    <li><span style="color: #60a5fa; font-weight: 600;">CSV Files</span> (.csv) - Comma-separated values</li>
                    <li><span style="color: #60a5fa; font-weight: 600;">Excel Files</span> (.xlsx) - Microsoft Excel spreadsheets</li>
                </ul>
                <p style="margin-top: 15px;">Our development roadmap includes support for JSON, SQL databases, and direct API connections in upcoming releases.</p>
                </div>
            """, unsafe_allow_html=True)
        
        with st.expander("Troubleshooting Common Issues"):
            st.markdown("""
                <div class="description-text">
                <p><strong class="highlight-text">File Upload Problems</strong></p>
                <ul style="line-height: 1.8; padding-left: 20px; margin-bottom: 20px;">
                    <li>Ensure your file is in CSV or Excel format with a valid extension</li>
                    <li>Check that the file size is under 200MB</li>
                    <li>Verify that your file is not corrupted by opening it in another application first</li>
                </ul>
                
                <p><strong class="highlight-text">Processing Delays</strong></p>
                <ul style="line-height: 1.8; padding-left: 20px; margin-bottom: 20px;">
                    <li>Large datasets (>100,000 rows) may require additional processing time</li>
                    <li>Files with many columns (>50) might take longer to analyze</li>
                    <li>Complex data types or mixed formats can extend processing duration</li>
                </ul>
                
                <p><strong class="highlight-text">Visualization Issues</strong></p>
                <ul style="line-height: 1.8; padding-left: 20px;">
                    <li>Ensure your dataset contains sufficient numeric columns for statistical analysis</li>
                    <li>Check for missing values that might affect visualization quality</li>
                    <li>Consider using a more balanced dataset if your data is heavily skewed</li>
                </ul>
                </div>
            """, unsafe_allow_html=True)
        
        with st.expander("Contact Support"):
            st.markdown("""
                <div class="description-text">
                <p>Our support team is ready to help with any questions or issues you encounter:</p>
                
                <div style="margin-top: 20px; margin-bottom: 20px; padding: 20px; background: rgba(59, 130, 246, 0.1); border-radius: 10px; border-left: 3px solid var(--primary-blue);">
                    <p style="margin-bottom: 10px;">📧 Email Support:</strong> support@devaai.com</p>
                    <p style="margin-bottom: 10px;">🌐 Support Portal:</strong> www.devaai.com/support</p>
                    <p style="margin-bottom: 0;">💬 Live Chat:</strong> Available on our website during business hours (9AM-6PM IST)</p>
                </div>
                
                <p>We aim to respond to all inquiries within 24 hours, with priority support available for enterprise customers.</p>
                </div>
            """, unsafe_allow_html=True)

        # Footer
        st.markdown("""
            <div class="footer">
                © 2025 DEVA AI - Transforming Data into Insights | <a href="#">Privacy Policy</a> | <a href="#">Terms of Service</a>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()