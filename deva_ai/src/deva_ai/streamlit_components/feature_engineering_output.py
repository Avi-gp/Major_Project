import pandas as pd
import streamlit as st
import os
from utils import parse_json_safely, extract_feature_engineering_info

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

        # Dataset Overview Section with cards - updated to show accurate column counts
        if "original_shape" in result and "final_shape" in result:
            col1, col2, col3 = st.columns([2,2,2])
            with col1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Original Dataset</div>
                        <div class="metric-value">{result["original_shape"][0]} × {result["original_shape"][1]}</div>
                        <div class="metric-label">(rows × columns)</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            with col2:
                # Calculate actual engineered column count
                engineered_cols = len(result.get("numerical_features", [])) + len(result.get("categorical_features", []))
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Engineered Features</div>
                        <div class="metric-value">{engineered_cols}</div>
                        <div class="metric-label">Total Features</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            with col3:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Final Dataset</div>
                        <div class="metric-value">{result["final_shape"][0]} × {engineered_cols}</div>
                        <div class="metric-label">(rows × columns)</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

        # Feature Distribution Section - using a dataframe
        st.markdown('<h3 class="highlight-text">Feature Distribution</h3>', unsafe_allow_html=True)

        # Create a dataframe for feature distribution
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

        # Engineering Methods Applied
        st.markdown('<h3 class="highlight-text">Engineering Methods Applied</h3>', unsafe_allow_html=True)
        method_tabs = st.tabs(["Scaling Methods", "Encoding Methods"])

        with method_tabs[0]:
            if "scaling_methods_applied" in result and result["scaling_methods_applied"]:
                scaling_df = pd.DataFrame([
                    {"Feature": k, "Method": v} 
                    for k, v in result["scaling_methods_applied"].items()
                ])
                st.dataframe(scaling_df, use_container_width=True)
            else:
                st.info("No scaling methods were applied")
                
        with method_tabs[1]:
            if "encoding_methods_applied" in result and result["encoding_methods_applied"]:
                encoding_df = pd.DataFrame([
                    {"Feature": k, "Method": v} 
                    for k, v in result["encoding_methods_applied"].items()
                ])
                st.dataframe(encoding_df, use_container_width=True)
            else:
                st.info("No encoding methods were applied")

        # Engineered Data Preview
        if "engineered_preview" in result and result["engineered_preview"]:
            st.markdown("### Engineered Data Preview")
            preview_df = pd.DataFrame(result["engineered_preview"])
            st.dataframe(preview_df, use_container_width=True)
            
        # Download Button
        if "engineered_file_path" in result and os.path.exists(result["engineered_file_path"]):
            file_path = result["engineered_file_path"]
            file_name = os.path.basename(file_path)
            
            with open(file_path, "rb") as file:
                st.download_button(
                    label="📥 Download Engineered Dataset",
                    data=file.read(),
                    file_name=file_name,
                    mime="text/csv" if file_name.endswith('.csv') else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Download the dataset with all engineering steps applied"
                )
        
        return True
        
    except Exception as e:
        st.error(f"Error displaying feature engineering results: {str(e)}")
        with st.expander("See detailed error information"):
            st.exception(e)
            st.code(raw_content, language="json")
        return False