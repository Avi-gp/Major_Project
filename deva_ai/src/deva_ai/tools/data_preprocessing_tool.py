# data_preprocessing_tool.py
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import os
import logging
import json
import re
from datetime import datetime

class PreprocessingInput(BaseModel):
    """Input schema for DataPreprocessingTool that uses file path."""
    file_path: str = Field(..., description="Absolute path to the uploaded file.")
    file_name: str = Field(..., description="The name of the uploaded file.")

class DataPreprocessingTool(BaseTool):
    name: str = "Data Preprocessing Tool"
    description: str = "Clean and preprocess datasets by handling missing values, outliers, duplicates, data types, and more"
    args_schema: Type[BaseModel] = PreprocessingInput

    def _run(self, file_path: str, file_name: str) -> str:
        """
        Process the file from a file path and perform comprehensive data preprocessing.
        
        Args:
            file_path (str): Absolute path to the file
            file_name (str): Name of the uploaded file
            
        Returns:
            str: JSON string containing preprocessing results
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found at path: {file_path}")
            
            # Read file based on extension
            file_extension = os.path.splitext(file_name)[1].lower()
            
            if file_extension == '.csv':
                df = pd.read_csv(file_path)
            elif file_extension == '.xlsx':
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Store original shape for comparison
            original_shape = df.shape
            original_missing = df.isna().sum().sum()
            
            # Get initial column dtypes
            initial_dtypes = {str(col): str(dtype) for col, dtype in df.dtypes.items()}
            
            # Execute preprocessing steps
            df, preprocessing_steps = self._preprocess_data(df)
            
            # Generate a new file path for the preprocessed data
            preprocessed_dir = os.path.join(os.path.dirname(file_path), "preprocessed")
            os.makedirs(preprocessed_dir, exist_ok=True)
            
            file_name_no_ext = os.path.splitext(file_name)[0]
            preprocessed_file_path = os.path.join(
                preprocessed_dir, 
                f"{file_name_no_ext}_preprocessed{file_extension}"
            )
            
            # Save preprocessed file
            if file_extension == '.csv':
                df.to_csv(preprocessed_file_path, index=False)
            elif file_extension == '.xlsx':
                df.to_excel(preprocessed_file_path, index=False)
            
            # Generate preview and final stats
            preview_rows = min(5, len(df))
            final_dtypes = {str(col): str(dtype) for col, dtype in df.dtypes.items()}
            
            type_changes = {
                col: {"original": initial_dtypes[col], "new": final_dtypes[col]}
                for col in final_dtypes if col in initial_dtypes and initial_dtypes[col] != final_dtypes[col]
                }
                
            # Create result dictionary with numpy values converted to Python standard types
            result = {
                "message": "Data preprocessing completed successfully",
                "original_shape": [int(original_shape[0]), int(original_shape[1])],
                "final_shape": [int(df.shape[0]), int(df.shape[1])],
                "original_missing_values": int(original_missing),
                "final_missing_values": int(df.isna().sum().sum()),
                "duplicates_removed": int(preprocessing_steps["duplicates_removed"]),
                "columns_dropped": preprocessing_steps["columns_dropped"],
                "outliers_handled": int(preprocessing_steps["outliers_handled"]),
                "datetime_columns": preprocessing_steps["datetime_columns"],
                "preprocessing_steps": self._convert_numpy_types(preprocessing_steps),
                "preview": df.head(preview_rows).to_dict('records'),
                "columns": df.columns.tolist(),
                "columns_dtypes": final_dtypes,
                "column_type_changes": type_changes,
                "preprocessed_file_path": preprocessed_file_path
            }
            
            # Return as a clean JSON string
            return json.dumps(result, default=self._json_serializer)
            
        except Exception as e:
            logging.error(f"Error preprocessing file: {str(e)}")
            error_result = {
                "error": str(e),
                "message": f"Failed to preprocess file at {file_path}"
            }
            return json.dumps(error_result)
    
    def _convert_numpy_types(self, obj):
        """
        Recursively convert numpy types to native Python types for JSON serialization.
        
        Args:
            obj: The object to convert
            
        Returns:
            Object with numpy types converted to native Python types
        """
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self._convert_numpy_types(obj.tolist())
        elif isinstance(obj, (np.bool_, np.bool)):
            return bool(obj)
        else:
            return obj
            
    def _json_serializer(self, obj):
        """
        Custom JSON serializer for objects not serializable by default json code.
        
        Args:
            obj: The object to serialize
            
        Returns:
            A JSON serializable version of the object
        """
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, np.bool)):
            return bool(obj)
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def _preprocess_data(self, df):
        """
        Perform comprehensive data preprocessing on the DataFrame.
        
        Args:
            df (pd.DataFrame): The DataFrame to preprocess
            
        Returns:
            tuple: (preprocessed_df, preprocessing_stats_dict)
        """
        preprocessing_steps = {
            "duplicates_removed": 0,
            "columns_dropped": [],
            "outliers_handled": 0,
            "dtypes_converted": [],
            "missing_values_handled": 0,
            "categorical_columns_cleaned": [],
            "datetime_columns": []
        }
        
        # 1. Data Type Conversion for numeric columns stored as objects
        df, converted_columns = self._convert_numeric_columns(df)
        preprocessing_steps["dtypes_converted"] = converted_columns
        
        # 2. Detect and Remove Duplicate Records
        original_rows = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        preprocessing_steps["duplicates_removed"] = original_rows - len(df)
        
        # 3. Identify and Drop Irrelevant Columns (ID columns, constant columns, etc.)
        df, dropped_columns = self._drop_irrelevant_columns(df)
        preprocessing_steps["columns_dropped"] = dropped_columns
        
        # 4. Clean Special Characters in Categorical Columns
        df, cleaned_columns = self._clean_categorical_columns(df)
        preprocessing_steps["categorical_columns_cleaned"] = cleaned_columns
        
        # 5. Advanced Type Inference for Datetime Columns
        df, datetime_columns = self._infer_datetime_columns(df)
        preprocessing_steps["datetime_columns"] = datetime_columns
        
        # 6. Handle Missing Values using appropriate imputation strategies
        df, missing_values_handled = self._handle_missing_values(df)
        preprocessing_steps["missing_values_handled"] = missing_values_handled
        
        # 7 & 8. Outlier Detection and Treatment using IQR method and clipping
        df, outliers_handled = self._handle_outliers(df)
        preprocessing_steps["outliers_handled"] = outliers_handled
        
        return df, preprocessing_steps
    
    def _convert_numeric_columns(self, df):
        """Convert object columns to numeric where possible."""
        converted_columns = []
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column can be converted to numeric
                try:
                    # Try to convert to numeric, coerce errors
                    numeric_series = pd.to_numeric(df[col], errors='coerce')
                    
                    # If >80% of values converted successfully, keep the conversion
                    non_null_pct = numeric_series.notna().mean()
                    if non_null_pct > 0.8:
                        df[col] = numeric_series
                        converted_columns.append(col)
                except:
                    pass
        
        return df, converted_columns
    
    def _drop_irrelevant_columns(self, df):
        """Identify and drop irrelevant columns."""
        dropped_columns = []
        
        # 1. Drop columns with high missing values (>75%)
        missing_threshold = 0.75
        missing_percentages = df.isnull().mean()
        high_missing_cols = missing_percentages[missing_percentages > missing_threshold].index.tolist()
        
        for col in high_missing_cols:
            if col in df.columns:  # Check if column still exists
                df = df.drop(col, axis=1)
                dropped_columns.append(f"{col} (high missing)")
        
        # 2. Drop columns that have only one unique value (constant/zero variance columns)
        zero_variance_cols = [col for col in df.columns if df[col].nunique() <= 1]
        for col in zero_variance_cols:
            if col in df.columns:  # Check if column still exists
                df = df.drop(col, axis=1)
                dropped_columns.append(f"{col} (constant)")
        
        # 3. Drop columns that have unique values for each row (likely ID columns)
        unique_cols = [col for col in df.columns if df[col].nunique() == len(df)]
        for col in unique_cols:
            if col in df.columns:  # Check if column still exists
                df = df.drop(col, axis=1)
                dropped_columns.append(f"{col} (unique/ID)")
        
        # 4. Additional check for columns with ID-like names and high cardinality
        for col in df.columns:
            # Skip if already dropped
            if col not in df.columns:
                continue
                
            # Check if column name contains 'id' and has high cardinality
            if (('id' in col.lower() or 'key' in col.lower() or 'uuid' in col.lower()) and 
                    df[col].nunique() > 0.9 * len(df)):
                df = df.drop(col, axis=1)
                dropped_columns.append(f"{col} (likely ID)")
        
        return df, dropped_columns
    
    def _clean_categorical_columns(self, df):
        """Clean categorical columns by removing spaces and special characters."""
        cleaned_columns = []
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Remove leading/trailing spaces
                if df[col].astype(str).str.strip().ne(df[col].astype(str)).any():
                    df[col] = df[col].astype(str).str.strip()
                    cleaned_columns.append(col)
                
                # Remove special characters if needed
                if df[col].astype(str).str.contains('[^\w\s]').any():
                    df[col] = df[col].astype(str).str.replace(r'[^\w\s]', '', regex=True)
                    if col not in cleaned_columns:
                        cleaned_columns.append(col)
        
        return df, cleaned_columns
    
    def _infer_datetime_columns(self, df):
        """Detect and convert columns with datetime format."""
        datetime_columns = []
        
        # Common datetime patterns to check
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
            r'\d{2}\.\d{2}\.\d{4}',  # DD.MM.YYYY
            r'\w+ \d{1,2}, \d{4}'   # Month DD, YYYY
        ]
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column contains dates
                sample = df[col].dropna().astype(str).sample(min(100, len(df))).tolist()
                
                date_match = False
                for pattern in date_patterns:
                    if any(re.search(pattern, str(val)) for val in sample):
                        date_match = True
                        break
                
                if date_match:
                    try:
                        # Try parsing the dates
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        datetime_columns.append(col)
                    except:
                        pass
        
        return df, datetime_columns
    
    def _handle_missing_values(self, df):
        """Handle missing values based on column data type."""
        missing_values_handled = 0
        
        for col in df.columns:
            # Get count of missing values in this column
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # For numeric columns, use median imputation
                    df[col] = df[col].fillna(df[col].median())
                    missing_values_handled += missing_count
                elif pd.api.types.is_datetime64_dtype(df[col]):
                    # For datetime columns, use forward fill or drop
                    if missing_count / len(df) < 0.2:  # If less than 20% missing
                        df[col] = df[col].fillna(method='ffill')
                        missing_values_handled += missing_count
                else:
                    # For categorical/string columns, fill with mode
                    mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
                    df[col] = df[col].fillna(mode_val)
                    missing_values_handled += missing_count
        
        return df, missing_values_handled
    
    def _handle_outliers(self, df):
        """Detect and handle outliers using IQR method."""
        outliers_handled = 0
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 5:
                # Calculate IQR
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define outlier bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Count outliers
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_count = outlier_mask.sum()
                
                if outlier_count > 0:
                    # Clip the outliers to the bounds
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                    outliers_handled += outlier_count
        
        return df, outliers_handled