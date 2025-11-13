
import os 
import pandas as pd 
import streamlit as st 
from utils import extract_preprocessing_info,parse_json_safely, extract_value

# Data Preprocessing Output Extraction 

def display_preprocessing_results(preprocessing_data):
    """
    Display the preprocessing task results in a dataset-agnostic way
    
    Args:
        preprocessing_data: The task output for preprocessing task
    """
    try:
        # Display preprocessing results with improved styling
        st.markdown('<h2 class="subheader">🔍 Data Preprocessing Results</h2>', unsafe_allow_html=True)
        
        # Extract preprocessing data from raw output
        raw_content = preprocessing_data.raw
        
        # Try parsing as JSON first
        result = parse_json_safely(raw_content)
        
        # If parsing didn't produce the expected fields, fall back to regex
        expected_keys = ["original_shape", "final_shape", "original_missing_values", "missing_values_handled"]
        if not result or not any(key in result for key in expected_keys):
            result = extract_preprocessing_info(raw_content)
        
        # Get preprocessed file information
        preprocessed_file_path = result.get("preprocessed_file_path", "")
        preprocessed_file_content = None
        preprocessed_file_name = None
        preprocessed_file_type = None
        
        # Read file content if it exists
        if preprocessed_file_path and os.path.exists(preprocessed_file_path):
            file_extension = os.path.splitext(preprocessed_file_path)[1].lower()
            file_type = "text/csv" if file_extension == ".csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            file_name = os.path.basename(preprocessed_file_path)
            
            with open(preprocessed_file_path, "rb") as file:
                preprocessed_file_content = file.read()
                
            preprocessed_file_name = file_name
            preprocessed_file_type = file_type
        
        # Create cards for key metrics
        with st.container():
            st.markdown('<h3 class="highlight-text">Preprocessing Summary</h3>', unsafe_allow_html=True)
            
            # Extract key metrics with fallbacks
            original_rows = extract_value(result, "original_shape", 0, 0)
            original_cols = extract_value(result, "original_shape", 1, 0)
            final_rows = extract_value(result, "final_shape", 0, 0)
            final_cols = extract_value(result, "final_shape", 1, 0)
            rows_removed = max(0, original_rows - final_rows)  # Prevent negative values
            cols_removed = max(0, original_cols - final_cols)  # Prevent negative values
            duplicates = extract_value(result, "duplicates_removed", None, 0)
            outliers = extract_value(result, "outliers_handled", None, 0)
            original_missing = extract_value(result, "original_missing_values", None, 0)
            final_missing = extract_value(result, "final_missing_values", None, 0)
            missing_values_h = extract_value(result,"missing_values_handled",None , 0)
            missing_fixed = max(0, missing_values_h)  # Prevent negative values
            
            # Display metrics in a row of 5 columns to include column count
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value">{final_rows}</div>
                        <div class="metric-label">Rows ({rows_removed} removed)</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value">{final_cols}</div>
                        <div class="metric-label">Columns ({cols_removed} removed)</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            with col3:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value">{duplicates}</div>
                        <div class="metric-label">Duplicates Removed</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            with col4:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value">{outliers}</div>
                        <div class="metric-label">Outliers Handled</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            with col5:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value">{missing_fixed}</div>
                        <div class="metric-label">Missing Values Fixed</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        
        # Handle all other result sections generically
        sections = [
            {"key": "transformations_applied", "title": "Transformations Applied", "columns": ["Transformation"]},
            {"key": "columns_dropped", "title": "Columns Dropped", "columns": ["Column", "Reason"]},
            {"key": "datetime_columns", "title": "Detected Date/Time Columns", "columns": ["Column"]}
        ]
        
        for section in sections:
            key = section["key"]
            if key in result and result[key]:
                st.markdown(f'<h3 class="highlight-text">{section["title"]}</h3>', unsafe_allow_html=True)
                
                data = result[key]
                df_data = []
                
                if isinstance(data, list):
                    if len(section["columns"]) == 1:
                        # Single column display (e.g., transformations, datetime columns)
                        for item in data:
                            if isinstance(item, dict) and section["columns"][0].lower() in map(str.lower, item.keys()):
                                # Find the matching key regardless of case
                                matching_key = next(k for k in item.keys() if k.lower() == section["columns"][0].lower())
                                df_data.append({section["columns"][0]: item[matching_key]})
                            else:
                                df_data.append({section["columns"][0]: str(item)})
                    elif len(section["columns"]) > 1 and key == "columns_dropped":
                        # Multiple column display (e.g., columns dropped with reasons)
                        for item in data:
                            if isinstance(item, dict) and "column" in item and "reason" in item:
                                df_data.append({
                                    "Column": item["column"],
                                    "Reason": item["reason"]
                                })
                            elif isinstance(item, str):
                                # Try to parse "column_name (reason)" format
                                parts = item.split(" (", 1)
                                if len(parts) > 1 and parts[1].endswith(")"):
                                    df_data.append({
                                        "Column": parts[0], 
                                        "Reason": parts[1].rstrip(")")
                                    })
                                else:
                                    df_data.append({"Column": item, "Reason": "N/A"})
                
                # Display as dataframe if we have data
                if df_data:
                    st.dataframe(pd.DataFrame(df_data), width='stretch')
                else:
                    # Fallback to string representation
                    st.write(", ".join(str(item) for item in data) if isinstance(data, list) else str(data))
        
        # Infer type conversions 
        if "column_type_changes" in result:
            st.markdown('<h3 class="highlight-text">Column Type Conversions</h3>', unsafe_allow_html=True)
            
            conversion_data = []
            type_changes = result["column_type_changes"]
            
            for column, change in type_changes.items():
                conversion_data.append({
                    "Column": column,
                    "Original Type": change.get("original", "N/A"),
                    "New Type": change.get("new", "N/A"),
                    "Conversion Description": f"The column {column} type is converted from {change.get('original', 'unknown')} to {change.get('new', 'unknown')}"
                })
            
            if conversion_data:
                conversion_df = pd.DataFrame(conversion_data)
                st.dataframe(conversion_df, width='stretch')
            else:
                st.write("No type conversions detected.")

        # If dtypes_converted is present (fallback method)
        elif "dtypes_converted" in result:
            st.markdown('<h3 class="highlight-text">Column Type Conversions</h3>', unsafe_allow_html=True)
            
            conversion_data = []
            dtypes_converted = result["dtypes_converted"]
            
            if isinstance(dtypes_converted, list):
                for change in dtypes_converted:
                    conversion_data.append({
                        "Column": change.get("column", "N/A"),
                        "Original Type": change.get("from_type", "N/A"),
                        "New Type": change.get("to_type", "N/A"),
                        "Conversion Description": f"The column {change.get('column', 'unknown')} type is converted from {change.get('from_type', 'unknown')} to {change.get('to_type', 'unknown')}"
                    })
            elif isinstance(dtypes_converted, dict):
                for column, change in dtypes_converted.items():
                    conversion_data.append({
                        "Column": column,
                        "Original Type": change.get("original", "N/A"),
                        "New Type": change.get("new", "N/A"),
                        "Conversion Description": f"The column {column} type is converted from {change.get('original', 'unknown')} to {change.get('new', 'unknown')}"
                    })
            
            if conversion_data:
                conversion_df = pd.DataFrame(conversion_data)
                st.dataframe(conversion_df, width='stretch')
            else:
                st.write("No type conversions detected.")
        
        # Column data types after preprocessing - handle generically
        if "columns_dtypes" in result and result["columns_dtypes"]:
            st.markdown('<h3 class="highlight-text">Column Data Types</h3>', unsafe_allow_html=True)
            
            dtypes = result["columns_dtypes"]
            dtype_data = []
            
            if isinstance(dtypes, dict):
                dtype_data = [{"Column": col, "Data Type": dtype} for col, dtype in dtypes.items()]
            elif isinstance(dtypes, list):
                for item in dtypes:
                    if isinstance(item, dict) and "column" in item and "type" in item:
                        dtype_data.append({
                            "Column": item["column"],
                            "Data Type": item["type"]
                        })
                    elif isinstance(item, str):
                        # Try to parse "column_name: type" format
                        parts = item.split(":", 1)
                        if len(parts) >= 2:
                            dtype_data.append({
                                "Column": parts[0].strip(),
                                "Data Type": parts[1].strip()
                            })
            
            if dtype_data:
                dtype_df = pd.DataFrame(dtype_data)
                
                # Apply custom formatting - highlight special types
                def highlight_datetime(val):
                    if 'date' in str(val).lower() or 'time' in str(val).lower():
                        return 'background-color: #2d4263; color: white'
                    return ''
                
                # Display styled dataframe - updated to use map instead of applymap
                st.dataframe(dtype_df.style.map(highlight_datetime, subset=['Data Type']), 
                           width='stretch')
            else:
                st.write("Data type information not available in expected format.")
        
        # Data Preview after preprocessing
        if "dataset_preview" in result and result["dataset_preview"]:
            st.markdown('<h3 class="highlight-text">Preprocessed Data Preview</h3>', unsafe_allow_html=True)
            
            # Handle different preview formats
            preview_data = result["dataset_preview"]
            
            try:
                if isinstance(preview_data, list):
                    if preview_data and isinstance(preview_data[0], dict):
                        # List of dictionaries
                        preprocessed_preview = pd.DataFrame(preview_data)
                    elif preview_data and isinstance(preview_data[0], list):
                        # List of lists (convert to dataframe with column names if available)
                        columns = result.get("columns", [])
                        if not columns and "columns_dtypes" in result:
                            columns = list(result["columns_dtypes"].keys())
                        
                        if columns and len(columns) == len(preview_data[0]):
                            preprocessed_preview = pd.DataFrame(preview_data, columns=columns)
                        else:
                            # Create default column names
                            preprocessed_preview = pd.DataFrame(preview_data)
                    else:
                        # Fall back to a simple dataframe
                        preprocessed_preview = pd.DataFrame({"Value": [str(item) for item in preview_data]})
                else:
                    # If preview is not a list, create a simple dataframe
                    preprocessed_preview = pd.DataFrame({"Preview": [str(preview_data)]})
                
                st.dataframe(preprocessed_preview, width='stretch')
                
            except Exception as preview_error:
                st.warning(f"Could not display preview data: {preview_error}")
                st.write("Raw preview data:", preview_data)

        # Handle download button for preprocessed file
        if preprocessed_file_content is not None:
            st.download_button(
                label="📥 Download Preprocessed Data",
                data=preprocessed_file_content,
                file_name=preprocessed_file_name,
                mime=preprocessed_file_type,
                key="download-preprocessed",
                help="Download the cleaned and preprocessed dataset"
            )
        
        return True
    
    except Exception as e:
        st.error(f"Error displaying preprocessing results: {str(e)}")
        # More detailed error information
        with st.expander("See detailed error information"):
            st.exception(e)
            st.code(preprocessing_data.raw, language="json")
        return False