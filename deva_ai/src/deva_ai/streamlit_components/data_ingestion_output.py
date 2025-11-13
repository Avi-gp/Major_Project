import re
import json 
import pandas as pd 
import streamlit as st 
from utils import extract_data_with_regex, parse_json_safely
# Data Ingestion Output Extraction

def display_data_ingestion_results(crew_output):
    """
    Display the data ingestion task results in a dataset-agnostic way
    
    Args:
        crew_output: The output from the CrewAI run
    """
    try:
        # Extract task output for data ingestion (task 0)
        if hasattr(crew_output, 'tasks_output') and len(crew_output.tasks_output) > 0:
            task_data = crew_output.tasks_output[0]
            
            # Try parsing as JSON first
            result = parse_json_safely(task_data.raw)
            
            # If parsing didn't produce the expected fields, fall back to regex
            if not result or not any(key in result for key in ["dataset_shape", "preview", "dtypes"]):
                result = extract_data_with_regex(task_data.raw)
            
            # Display results in a structured format
            st.markdown('<h2 class="subheader">📊 Data Ingestion Results</h2>', unsafe_allow_html=True)
            
            # Dataset Overview
            if "dataset_shape" in result and result["dataset_shape"]:
                st.markdown('<h3 class="highlight-text">Dataset Overview</h3>', unsafe_allow_html=True)
                
                # Handle different formats of dataset_shape
                shape_data = result["dataset_shape"]
                rows = 0
                cols = 0
                
                if isinstance(shape_data, list) and len(shape_data) >= 2:
                    rows, cols = shape_data[0], shape_data[1]
                elif isinstance(shape_data, dict) and "rows" in shape_data and "columns" in shape_data:
                    rows, cols = shape_data["rows"], shape_data["columns"]
                elif isinstance(shape_data, str):
                    # Try to extract numbers from string like "rows: 1000, columns: 15"
                    rows_match = re.search(r'rows:?\s*(\d+)', shape_data, re.IGNORECASE)
                    cols_match = re.search(r'columns:?\s*(\d+)', shape_data, re.IGNORECASE)
                    
                    if rows_match:
                        rows = int(rows_match.group(1))
                    if cols_match:
                        cols = int(cols_match.group(1))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-value">{rows}</div>
                            <div class="metric-label">Rows</div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                with col2:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-value">{cols}</div>
                            <div class="metric-label">Columns</div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
            
            # Data Preview
            if "preview" in result and result["preview"]:
                st.markdown('<h3 class="highlight-text">Data Preview</h3>', unsafe_allow_html=True)
                
                preview_data = result["preview"]
                
                # Handle different formats of preview data
                if isinstance(preview_data, list):
                    if preview_data and isinstance(preview_data[0], dict):
                        # List of dictionaries
                        preview_df = pd.DataFrame(preview_data)
                    elif preview_data and isinstance(preview_data[0], list):
                        # List of lists (convert to dataframe with default column names)
                        preview_df = pd.DataFrame(preview_data)
                    else:
                        # Fall back to a simple dataframe
                        preview_df = pd.DataFrame({"Value": preview_data})
                else:
                    # If preview is not a list, create a simple dataframe
                    preview_df = pd.DataFrame({"Preview": [str(preview_data)]})
                
                st.dataframe(preview_df, width='stretch')
            
            # Column Information
            if "dtypes" in result and result["dtypes"]:
                st.markdown('<h3 class="highlight-text">Column Information</h3>', unsafe_allow_html=True)
                
                dtypes_data = result["dtypes"]
                col_info = []
                
                # Handle different formats of dtypes data
                if isinstance(dtypes_data, dict):
                    for col, dtype in dtypes_data.items():
                        col_info.append({"Column": col, "Type": dtype})
                elif isinstance(dtypes_data, list):
                    for item in dtypes_data:
                        if isinstance(item, dict) and "column" in item and "type" in item:
                            col_info.append({"Column": item["column"], "Type": item["type"]})
                        elif isinstance(item, str):
                            # Try to extract column and type from string
                            parts = item.split(":")
                            if len(parts) >= 2:
                                col_info.append({"Column": parts[0].strip(), "Type": parts[1].strip()})
                
                if col_info:
                    st.dataframe(pd.DataFrame(col_info), width='stretch')
                else:
                    st.write("Column type information not available in expected format.")
                
            return True
        else:
            st.warning("No data ingestion results found in the crew output.")
            return False
            
    except Exception as e:
        st.error(f"Error displaying data ingestion results: {str(e)}")
        # Provide more detailed error information in an expander
        with st.expander("See detailed error information"):
            st.exception(e)
            st.code(crew_output.tasks_output[0].raw if hasattr(crew_output, 'tasks_output') and len(crew_output.tasks_output) > 0 else "No raw data available", language="json")
        return False
