import re
import json
import pandas as pd
import streamlit as st
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import extract_value, parse_json_safely, extract_feature_engineering_info
import plotly.express as px
import plotly.graph_objects as go

def display_feature_engineering_results(feature_engineering_data):
    """
    Display the feature engineering task results in a structured format
    
    Args:
        feature_engineering_data: The task output for feature engineering task
    """
    try:
        # Extract data from raw output
        raw_content = feature_engineering_data.raw
        
        # Try parsing as JSON first
        result = parse_json_safely(raw_content)
        
        # If parsing failed or missing key fields, try feature engineering specific extraction
        if not result or not any(key in result for key in ["original_shape", "numerical_features"]):
            result = extract_feature_engineering_info(raw_content)
            
        # Display results
        st.markdown('<h2 class="subheader">🔧 Feature Engineering Results</h2>', unsafe_allow_html=True)
        
        # Dataset Overview - Keep the original cards for shape information
        if "original_shape" in result and "final_shape" in result:
            col1, col2 = st.columns([2,2])
            with col1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Preprocessed Dataset</div>
                        <div class="metric-value">{result["original_shape"][0]} × {result["original_shape"][1]}</div>
                        <div class="metric-label">Rows × Columns</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Engineered Dataset</div>
                        <div class="metric-value">{result["final_shape"][0]} × {result["final_shape"][1]}</div>
                        <div class="metric-label">Rows × Columns</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

        # Feature Types - Changed to use a table for feature distribution
        st.markdown('<h3 class="highlight-text">Feature Distribution</h3>', unsafe_allow_html=True)

        # Create a dataframe with feature counts and lists
        feature_data = {}
        if "numerical_features" in result:
            feature_data["Numerical Features"] = [
                len(result['numerical_features']),
                ", ".join(result["numerical_features"])
            ]
            
        if "categorical_features" in result:
            feature_data["Categorical Features"] = [
                len(result['categorical_features']),
                ", ".join(result["categorical_features"])
            ]

        # Create and display the dataframe
        if feature_data:
            feature_df = pd.DataFrame(feature_data, index=["Count", "Features"]).transpose()
            st.dataframe(feature_df, use_container_width=True)

        # Updated CSS for containers
        st.markdown(
            """
            <style>
            [data-testid="stContainer"] {
                background-color: #f8f9fa;
                border-radius: 10px;
                padding: 1rem;
                margin: 0.5rem 0;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Feature Engineering Methods Applied
        st.markdown('<h3 class="highlight-text">Feature Engineering Methods</h3>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Scaling Methods", "Encoding Methods"])
        
        with tab1:
            if "scaling_methods_applied" in result and result["scaling_methods_applied"]:
                scaling_df = pd.DataFrame([
                    {"Feature": k, "Scaling Method": v} 
                    for k, v in result["scaling_methods_applied"].items()
                ])
                st.dataframe(scaling_df, use_container_width=True)
            else:
                st.info("No scaling methods were applied")
                
        with tab2:
            if "encoding_methods_applied" in result and result["encoding_methods_applied"]:
                encoding_df = pd.DataFrame([
                    {"Feature": k, "Encoding Method": v} 
                    for k, v in result["encoding_methods_applied"].items()
                ])
                st.dataframe(encoding_df, use_container_width=True)
            else:
                st.info("No encoding methods were applied")
        
        # Feature Analysis Results
        st.markdown('<h3 class="highlight-text">Feature Analysis</h3>', unsafe_allow_html=True)
        
        # Feature Importance
        if "feature_importance" in result and result["feature_importance"]:
            with st.expander("📊 Feature Importance Analysis", expanded=True):
                importance_data = result["feature_importance"]
                if isinstance(importance_data, dict) and "features" in importance_data:
                    features_dict = importance_data["features"]
                else:
                    features_dict = importance_data
                
                if features_dict:
                    # Create DataFrame for visualization
                    importance_df = pd.DataFrame([
                        {"Feature": k, "Importance": v} 
                        for k, v in features_dict.items()
                    ]).sort_values("Importance", ascending=False)
                    
                    # Display table
                    st.dataframe(importance_df, use_container_width=True)
                    
                    # Create bar plot
                    fig = px.bar(
                        importance_df,
                        x="Feature",
                        y="Importance",
                        title=f"Feature Importance Distribution (Target: {result.get('target_column', 'Unknown')})"
                    )
                    fig.update_layout(
                        xaxis_title="Features",
                        yaxis_title=f"Importance Score relative to {result.get('target_column', 'target')}",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Correlation Analysis - Moved outside expander and using seaborn
        if "high_correlation_pairs" in result and result["high_correlation_pairs"]:
            st.markdown("## 🔄 High Correlation Analysis")
            
            # Display correlation pairs table
            corr_df = pd.DataFrame(
                result["high_correlation_pairs"],
                columns=["Feature 1", "Feature 2", "Correlation"]
            )
            st.dataframe(corr_df, use_container_width=True)
            
            # Create correlation matrix for heatmap
            features = list(set(corr_df["Feature 1"].tolist() + corr_df["Feature 2"].tolist()))
            corr_matrix = pd.DataFrame(0.0, index=features, columns=features)  # Initialize with float
            
            # Fill correlation matrix
            for _, row in corr_df.iterrows():
                corr_val = float(row["Correlation"])  # Explicit conversion to float
                corr_matrix.loc[row["Feature 1"], row["Feature 2"]] = corr_val
                corr_matrix.loc[row["Feature 2"], row["Feature 1"]] = corr_val
            np.fill_diagonal(corr_matrix.values, 1.0)
            
            # Create heatmap using seaborn
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                corr_matrix,
                annot=True,
                cmap='coolwarm',
                fmt='.2f',
                square=True,
                cbar_kws={'label': 'Correlation'}
            )
            plt.title("Feature Correlation Heatmap")
            plt.tight_layout()
            
            st.pyplot(plt)

        # Low Variance Features
        if "variance_filtered_features" in result and result["variance_filtered_features"]:
            with st.expander("📉 Low Variance Features"):
                variance_df = pd.DataFrame(result["variance_filtered_features"])
                st.dataframe(variance_df, use_container_width=True)

        # Suggested Features to Drop
        if "suggested_features_to_drop" in result and result["suggested_features_to_drop"]:
            with st.expander("❌ Suggested Features to Drop", expanded=True):
                drops_df = pd.DataFrame(result["suggested_features_to_drop"])
                st.dataframe(drops_df, use_container_width=True)

        # Data Preview
        if "engineered_preview" in result and result["engineered_preview"]:
            st.markdown('<h3 class="highlight-text">Engineered Data Preview</h3>', unsafe_allow_html=True)
            preview_df = pd.DataFrame(result["engineered_preview"])
            st.dataframe(preview_df, use_container_width=True)
        
        # Use the same styled download button as preprocessing output
        if "engineered_file_path" in result and result["engineered_file_path"]:
            file_path = result["engineered_file_path"]
            if os.path.exists(file_path):
                with open(file_path, "rb") as file:
                    file_data = file.read()
                    
                file_name = os.path.basename(file_path)
                file_extension = os.path.splitext(file_path)[1].lower()
                file_type = "text/csv" if file_extension == ".csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                
                
                
                # Use callback to prevent re-execution
                def get_download_data():
                    return file_data
                    
                st.download_button(
                    label="📥 Download Engineered Dataset",
                    data=get_download_data(),
                    file_name=file_name,
                    mime=file_type,
                    key=f"download-engdata-{hash(file_path)}",
                    help="Download the engineered dataset"
                )
                
        return True
    
    except Exception as e:
        st.error(f"Error displaying feature engineering results: {str(e)}")
        with st.expander("See detailed error information"):
            st.exception(e)
            st.code(feature_engineering_data.raw, language="json")
        return False