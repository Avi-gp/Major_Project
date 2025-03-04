import json
import pandas as pd
import os
import re

def save_uploadedfile(uploaded_file):
    """Save uploaded file and return absolute file path"""
    try:
        # Create datasets directory in current working directory
        save_dir = os.path.join(os.getcwd(), "datasets")
        os.makedirs(save_dir, exist_ok=True)
        
        # Create absolute file path
        file_path = os.path.abspath(os.path.join(save_dir, uploaded_file.name))
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
            
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

#--------------------------------------------------------------------------------------------------------
def extract_data_with_regex(raw_text):
    """Extract data directly using regex when JSON parsing fails - dataset agnostic approach"""
    result = {}
    
    # Extract dataset shape
    shape_match = re.search(r'"dataset_shape":\s*\[(\d+),\s*(\d+)\]', raw_text)
    if shape_match:
        result["dataset_shape"] = [int(shape_match.group(1)), int(shape_match.group(2))]
    
    # Extract preview data in a generic way
    preview = []
    # Look for objects in the preview array
    preview_objects = re.findall(r'\{"([^"]+)":\s*("[^"]+"|\d+|null|true|false).*?\}', raw_text, re.DOTALL)
    
    seen_objects = set()  # Track objects we've already processed to avoid duplicates
    
    for match in preview_objects:
        # Get the first part of each object to identify it
        first_field = match[0]
        
        # Find the full object for this match
        object_pattern = r'\{([^{}]*"' + re.escape(first_field) + r'"[^{}]*)\}'
        object_match = re.search(object_pattern, raw_text)
        
        if object_match and object_match.group(0) not in seen_objects:
            object_str = object_match.group(0)
            seen_objects.add(object_str)
            
            # Clean and parse the object
            cleaned_obj = object_str.replace("'", '"')  # Replace single quotes with double quotes
            
            try:
                # Convert keys without quotes to proper JSON format
                cleaned_obj = re.sub(r'(\w+):', r'"\1":', cleaned_obj)
                # Fix trailing commas
                cleaned_obj = re.sub(r',\s*}', '}', cleaned_obj)
                
                row_data = json.loads(cleaned_obj)
                preview.append(row_data)
                
                if len(preview) >= 5:  # Limit to 5 rows
                    break
            except json.JSONDecodeError:
                # If JSON parsing fails, create a simple key-value object
                pairs = re.findall(r'"([^"]+)":\s*("[^"]+"|[\d\.]+|null|true|false)', object_str)
                row_data = {k: v.strip('"') if v.startswith('"') else v for k, v in pairs}
                preview.append(row_data)
                
                if len(preview) >= 5:  # Limit to 5 rows
                    break
    
    if preview:
        result["preview"] = preview
    
    # Extract dtypes in a generic way
    dtypes = {}
    dtypes_match = re.search(r'"dtypes":\s*\{(.*?)\}', raw_text, re.DOTALL)
    if dtypes_match:
        dtypes_text = dtypes_match.group(1)
        dtype_pairs = re.findall(r'"([^"]+)":\s*"([^"]+)"', dtypes_text)
        for col, dtype in dtype_pairs:
            dtypes[col] = dtype
    
    result["dtypes"] = dtypes
    
    return result


#------------------------------------------------------------------------------------#
def extract_preprocessing_info(raw_text):
    """Extract preprocessing information in a dataset-agnostic way"""
    result = {}
    
    # Extract dataset shapes
    original_shape_match = re.search(r'"original_shape":\s*\[(\d+),\s*(\d+)\]', raw_text)
    if original_shape_match:
        result["original_shape"] = [int(original_shape_match.group(1)), int(original_shape_match.group(2))]
    
    final_shape_match = re.search(r'"final_shape":\s*\[(\d+),\s*(\d+)\]', raw_text)
    if final_shape_match:
        result["final_shape"] = [int(final_shape_match.group(1)), int(final_shape_match.group(2))]
    
    # Extract other metrics
    for metric in ["original_missing_values", "final_missing_values", "duplicates_removed", "outliers_handled","missing_values_handled"]:
        metric_match = re.search(rf'"{metric}":\s*(\d+)', raw_text)
        if metric_match:
            result[metric] = int(metric_match.group(1))
    
    # Extract columns dropped
    columns_dropped_match = re.search(r'"columns_dropped":\s*\[(.*?)\]', raw_text, re.DOTALL)
    if columns_dropped_match:
        columns_str = columns_dropped_match.group(1)
        # Extract quoted strings from the list
        columns = re.findall(r'"([^"]+)"', columns_str)
        if columns:
            result["columns_dropped"] = columns
    
    # Extract datetime columns
    datetime_cols_match = re.search(r'"datetime_columns":\s*\[(.*?)\]', raw_text, re.DOTALL)
    if datetime_cols_match:
        datetime_str = datetime_cols_match.group(1)
        datetime_cols = re.findall(r'"([^"]+)"', datetime_str)
        if datetime_cols:
            result["datetime_columns"] = datetime_cols
    
    # Extract transformations applied
    transformations_match = re.search(r'"transformations_applied":\s*\[(.*?)\]', raw_text, re.DOTALL)
    if transformations_match:
        transformations_str = transformations_match.group(1)
        transformations = re.findall(r'"([^"]+)"', transformations_str)
        if transformations:
            result["transformations_applied"] = transformations
    
    # Extract original column dtypes
    original_dtypes_match = re.search(r'"dtypes":\s*\{(.*?)\}', raw_text, re.DOTALL)
    if original_dtypes_match:
        dtypes_str = original_dtypes_match.group(1)
        # Extract key-value pairs
        dtype_pairs = re.findall(r'"([^"]+)":\s*"([^"]+)"', dtypes_str)
        if dtype_pairs:
            result["dtypes"] = {col: dtype for col, dtype in dtype_pairs}
    
    # Extract column dtypes
    dtypes_match = re.search(r'"columns_dtypes":\s*\{(.*?)\}', raw_text, re.DOTALL)
    if dtypes_match:
        dtypes_str = dtypes_match.group(1)
        # Extract key-value pairs
        dtype_pairs = re.findall(r'"([^"]+)":\s*"([^"]+)"', dtypes_str)
        if dtype_pairs:
            result["columns_dtypes"] = {col: dtype for col, dtype in dtype_pairs}
    
    # Add type conversion extraction
    type_conversions_match = re.search(r'"column_type_changes":\s*(\{.*?\})', raw_text, re.DOTALL)
    if type_conversions_match:
        conversions_str = type_conversions_match.group(1)
        
        try:
            # Clean and parse the type conversions
            cleaned_str = preprocess_for_json(conversions_str)
            result["column_type_changes"] = json.loads(cleaned_str)
        except json.JSONDecodeError:
            # Fallback to regex extraction if JSON parsing fails
            conversions = {}
            conv_matches = re.findall(r'"([^"]+)":\s*\{([^}]+)\}', conversions_str)
            
            for column, type_info in conv_matches:
                # Extract original and new types
                original_match = re.search(r'"original":\s*"([^"]+)"', type_info)
                new_match = re.search(r'"new":\s*"([^"]+)"', type_info)
                
                if original_match and new_match:
                    conversions[column] = {
                        "original": original_match.group(1),
                        "new": new_match.group(1)
                    }
            
            if conversions:
                result["column_type_changes"] = conversions
    
    # Extract preview data in a more generic way
    preview_match = re.search(r'"dataset_preview":\s*(\[.*?\])', raw_text, re.DOTALL)
    if preview_match:
        preview_str = preview_match.group(1)
        # First attempt to parse as JSON
        try:
            cleaned_json = re.sub(r'(\w+):', r'"\1":', preview_str)
            cleaned_json = re.sub(r'\'', r'"', cleaned_json)
            cleaned_json = re.sub(r',\s*\]', ']', cleaned_json)
            
            preview_data = json.loads(cleaned_json)
            result["dataset_preview"] = preview_data
        except json.JSONDecodeError:
            # Fallback to regex extraction
            objects = []
            object_matches = re.findall(r'\{([^{}]*)\}', preview_str)
            for obj_str in object_matches:
                pairs = re.findall(r'"?([^":,\s]+)"?\s*:\s*("[^"]*"|[^",\s]+)', obj_str)
                if pairs:
                    obj = {}
                    for key, val in pairs:
                        if val.startswith('"') and val.endswith('"'):
                            obj[key] = val[1:-1]
                        elif val.lower() in ['true', 'false']:
                            obj[key] = val.lower() == 'true'
                        elif val.lower() == 'null':
                            obj[key] = None
                        else:
                            try:
                                obj[key] = float(val) if '.' in val else int(val)
                            except:
                                obj[key] = val
                    objects.append(obj)
            
            if objects:
                result["preview"] = objects
    
    # Extract file path
    file_path_match = re.search(r'"preprocessed_file_path":\s*"([^"]+)"', raw_text)
    if file_path_match:
        result["preprocessed_file_path"] = file_path_match.group(1)
    
    return result

#---------------------------------------------------------------------------------------------#
def preprocess_for_json(raw_content):
    """Preprocess raw content to ensure it is valid JSON"""
    # Replace tuples with lists (tuples use round brackets)
    raw_content = re.sub(r'\((.*?)\)', r'[\1]', raw_content)
    
    # Convert all keys without quotes to properly quoted JSON keys
    raw_content = re.sub(r'(\w+)(?=\s*:)', r'"\1"', raw_content)
    
    # Replace single quotes with double quotes
    raw_content = raw_content.replace("'", '"')
    
    # Remove trailing commas before closing brackets
    raw_content = re.sub(r',\s*\}', '}', raw_content)
    raw_content = re.sub(r',\s*\]', ']', raw_content)
    
    # Additional fixes for any other non-JSON-friendly content
    raw_content = raw_content.replace('...', '')
    
    return raw_content


#---------------------------------------------------------------#
def parse_json_safely(raw_text):
    """
    Attempt to parse JSON from raw text with progressive fallback strategies
    
    Args:
        raw_text (str): Raw text potentially containing JSON
        
    Returns:
        dict: Parsed JSON or empty dict if parsing fails
    """
    # First attempt: Try to find a complete JSON object
    json_pattern = r'(\{.*\})'
    json_match = re.search(json_pattern, raw_text, re.DOTALL)
    
    if json_match:
        json_str = json_match.group(1)
        
        # Try parsing as-is
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Clean up the string to make it valid JSON
        cleaned_json = preprocess_for_json(json_str)
        
        try:
            return json.loads(cleaned_json)
        except json.JSONDecodeError:
            pass
    
    # If no complete JSON found, try to extract key components
    result = {}
    
    # Look for common patterns that might indicate a JSON structure
    for key in ["dataset_shape", "preview", "dtypes", "original_shape", "final_shape", 
                "columns_dropped", "transformations_applied", "columns_dtypes","dataset_preview"]:
        
        # Look for array values
        array_pattern = rf'"{key}":\s*(\[.*?\])'
        array_match = re.search(array_pattern, raw_text, re.DOTALL)
        
        if array_match:
            array_str = array_match.group(1)
            cleaned_array = preprocess_for_json(array_str)
            
            try:
                result[key] = json.loads(cleaned_array)
            except:
                # If parsing fails, just store as string
                result[key] = array_str
        
        # Look for object values
        object_pattern = rf'"{key}":\s*(\{{.*?\}})'
        object_match = re.search(object_pattern, raw_text, re.DOTALL)
        
        if object_match:
            object_str = object_match.group(1)
            cleaned_object = preprocess_for_json(object_str)
            
            try:
                result[key] = json.loads(cleaned_object)
            except:
                # If parsing fails, just store as string
                result[key] = object_str
        
        # Look for simple string/number values
        value_pattern = rf'"{key}":\s*"([^"]*)"'
        value_match = re.search(value_pattern, raw_text)
        
        if value_match:
            result[key] = value_match.group(1)
        else:
            # Try numeric values
            num_pattern = rf'"{key}":\s*(\d+)'
            num_match = re.search(num_pattern, raw_text)
            
            if num_match:
                result[key] = int(num_match.group(1))

    # Add an additional check for nested dictionaries with specific keys
    nested_dict_keys = [
        "column_type_changes", 
        "dtypes_converted", 
        "type_conversions"
    ]

    for key in nested_dict_keys:
        nested_pattern = rf'"{key}":\s*(\{{.*?\}})'
        nested_match = re.search(nested_pattern, raw_text, re.DOTALL)
        
        if nested_match:
            nested_str = nested_match.group(1)
            cleaned_nested = preprocess_for_json(nested_str)
            
            try:
                nested_result = json.loads(cleaned_nested)
                result[key] = nested_result
            except:
                # If parsing fails, store as string or use regex extraction
                pass
        
    return result

#----------------------------------------------------------------------------------------------#

def extract_value(data_dict, key, index=None, default=None):
    """
    Safely extract a value from a dictionary, with improved handling for 
    different data formats
    
    Args:
        data_dict (dict): Dictionary to extract from
        key (str): Dictionary key to access
        index (int, optional): Index to access if value is a list
        default: Default value to return if extraction fails
        
    Returns:
        The extracted value or default value if extraction fails
    """
    try:
        if key not in data_dict:
            return default
            
        value = data_dict[key]
        
        # New handler for nested dictionaries with type conversion format
        if isinstance(value, dict):
            # Handle column type changes specifically
            if "original" in value and "new" in value:
                return f"{value.get('original', 'N/A')} → {value.get('new', 'N/A')}"
        
        # Existing handling remains the same
        if index is not None:
            if isinstance(value, list) and len(value) > index:
                return value[index]
            elif isinstance(value, dict) and "value" in value:
                return value["value"]
            elif isinstance(value, str):
                # Try to extract a number from a string
                number_match = re.search(r'\d+', value)
                if number_match:
                    return int(number_match.group(0))
        
        # If value is a string containing only a number, convert it
        if isinstance(value, str):
            if value.isdigit():
                return int(value)
            try:
                return float(value)
            except ValueError:
                pass
                
        return value
    except (TypeError, IndexError, KeyError):
        return default