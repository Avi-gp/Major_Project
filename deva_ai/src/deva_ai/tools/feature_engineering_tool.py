# feature_engineering_tool.py
from typing import Type, Optional
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import os
import logging
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder

class FeatureEngineeringInput(BaseModel):
    """Input schema for FeatureEngineeringTool that uses file path."""
    preprocessed_file_path: str = Field(..., description="Absolute path to the preprocessed file.")
    preprocessed_file_name: str = Field(..., description="The name of the preprocessed file.")

class FeatureEngineeringTool(BaseTool):
    name: str = "Feature Engineering Tool"
    description: str = "Transform preprocessed data by applying scaling techniques for numerical data and encoding methods for categorical data"
    args_schema: Type[BaseModel] = FeatureEngineeringInput

    def _run(self, preprocessed_file_path: str, preprocessed_file_name: str) -> str:
        """
        Process the preprocessed file and perform feature engineering.
        
        Args:
            preprocessed_file_path (str): Absolute path to the preprocessed file
            preprocessed_file_name (str): Name of the preprocessed file
            
        Returns:
            str: JSON string containing feature engineering results
        """
        try:
            # Check if file exists
            if not os.path.exists(preprocessed_file_path):
                raise FileNotFoundError(f"File not found at path: {preprocessed_file_path}")
            
            # Read file based on extension
            file_extension = os.path.splitext(preprocessed_file_name)[1].lower()
            
            if file_extension == '.csv':
                df = pd.read_csv(preprocessed_file_path)
            elif file_extension == '.xlsx':
                df = pd.read_excel(preprocessed_file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Store original shape for comparison
            original_shape = df.shape
            
            # Identify feature types
            numerical_features, categorical_features = self._identify_feature_types(df)
            
            # Execute feature engineering steps
            df, feature_engineering_stats = self._engineer_features(df, numerical_features, categorical_features)
            
            # Generate a new file path for the engineered data
            engineered_dir = os.path.join(os.path.dirname(os.path.dirname(preprocessed_file_path)), "engineered")
            os.makedirs(engineered_dir, exist_ok=True)
            
            file_name_no_ext = os.path.splitext(preprocessed_file_name)[0].replace("_preprocessed", "")
            engineered_file_path = os.path.join(
                engineered_dir, 
                f"{file_name_no_ext}_engineered{file_extension}"
            )
            
            # Save engineered file
            if file_extension == '.csv':
                df.to_csv(engineered_file_path, index=False)
            elif file_extension == '.xlsx':
                df.to_excel(engineered_file_path, index=False)
            
            # Generate preview and final stats
            preview_rows = min(5, len(df))
            
            # Create result dictionary with numpy values converted to Python standard types
            result = {
                "message": "Feature engineering completed successfully",
                "original_shape": [int(original_shape[0]), int(original_shape[1])],
                "final_shape": [int(df.shape[0]), int(df.shape[1])],
                "numerical_features": numerical_features,
                "categorical_features": categorical_features,
                "scaling_methods_applied": feature_engineering_stats["scaling_methods_applied"],
                "encoding_methods_applied": feature_engineering_stats["encoding_methods_applied"],
                "engineered_preview": df.head(preview_rows).to_dict('records'),
                "engineered_file_path": engineered_file_path,
                "engineered_file_name": os.path.basename(engineered_file_path)
            }
            
            # Return as a clean JSON string
            return json.dumps(result, default=self._json_serializer)
            
        except Exception as e:
            logging.error(f"Error engineering features: {str(e)}")
            error_result = {
                "error": str(e),
                "message": f"Failed to engineer features for file at {preprocessed_file_path}"
            }
            return json.dumps(error_result)
    
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
    
    def _identify_feature_types(self, df):
        """
        Identify numerical and categorical features in the DataFrame.
        
        Args:
            df (pd.DataFrame): The DataFrame to analyze
            
        Returns:
            tuple: (numerical_features_list, categorical_features_list)
        """
        numerical_features = []
        categorical_features = []
        
        for column in df.columns:
            # Check if column is datetime type
            if pd.api.types.is_datetime64_dtype(df[column]) or pd.api.types.is_datetime64_any_dtype(df[column]):
                # Datetime columns are handled separately and not included in feature lists
                continue
                
            # Check if column is numeric
            elif pd.api.types.is_numeric_dtype(df[column]):
                # Check if it's actually a categorical variable encoded as numeric
                unique_count = df[column].nunique()
                
                # If it has few unique values and they're integers, it might be categorical
                if unique_count <= 10 and pd.api.types.is_integer_dtype(df[column]):
                    # Check if values are evenly spaced
                    values = sorted(df[column].unique())
                    if len(values) >= 2:
                        # If values look like a range (e.g., 1,2,3,4), consider it categorical
                        categorical_features.append(column)
                    else:
                        numerical_features.append(column)
                else:
                    numerical_features.append(column)
            
            # Check if it's a categorical or object type
            elif pd.api.types.is_categorical_dtype(df[column]) or df[column].dtype == 'object':
                categorical_features.append(column)
        
        return numerical_features, categorical_features
    
    def _engineer_features(self, df, numerical_features, categorical_features):
        """
        Apply feature engineering to the DataFrame.
        
        Args:
            df (pd.DataFrame): The DataFrame to engineer
            numerical_features (list): List of numerical feature names
            categorical_features (list): List of categorical feature names
            
        Returns:
            tuple: (engineered_df, feature_engineering_stats_dict)
        """
        feature_engineering_stats = {
            "scaling_methods_applied": {},
            "encoding_methods_applied": {}
        }
        
        # Part 1: Numerical Feature Scaling
        df, scaling_stats = self._scale_numerical_features(df, numerical_features)
        feature_engineering_stats["scaling_methods_applied"] = scaling_stats
        
        # Part 2: Categorical Feature Encoding
        df, encoding_stats = self._encode_categorical_features(df, categorical_features)
        feature_engineering_stats["encoding_methods_applied"] = encoding_stats
        
        return df, feature_engineering_stats
    
    def _scale_numerical_features(self, df, numerical_features, round_decimals=4):
        """
        Scale numerical features using appropriate methods and round the results.
        
        Args:
            df (pd.DataFrame): DataFrame to process
            numerical_features (list): List of numerical column names
            round_decimals (int): Number of decimal places to round to
            
        Returns:
            tuple: (scaled_df, scaling_methods_dict)
        """
        scaling_methods = {}
        
        for column in numerical_features:
            # Skip if column doesn't exist
            if column not in df.columns:
                continue
                    
            # Skip columns with all NaN values
            if df[column].isna().all():
                continue
                    
            # Check distribution characteristics to select appropriate scaling method
            skewness = df[column].skew()
            has_outliers = self._detect_outliers(df[column])
            
            # Choose scaling method based on data characteristics
            if has_outliers:
                # Use Robust Scaler for data with outliers
                scaler = RobustScaler()
                scaled_values = scaler.fit_transform(df[[column]])
                df[column] = np.round(scaled_values, round_decimals)
                scaling_methods[column] = "robust_scaling"
                    
            elif abs(skewness) > 1.0:
                # Use Min-Max scaling for skewed data
                scaler = MinMaxScaler()
                scaled_values = scaler.fit_transform(df[[column]])
                df[column] = np.round(scaled_values, round_decimals)
                scaling_methods[column] = "minmax_scaling"
                    
            else:
                # Use Standardization for normally distributed data
                scaler = StandardScaler()
                scaled_values = scaler.fit_transform(df[[column]])
                df[column] = np.round(scaled_values, round_decimals)
                scaling_methods[column] = "standardization"
            
        return df, scaling_methods
    
    def _encode_categorical_features(self, df, categorical_features):
        """
        Encode categorical features using optimized methods:
        - One-hot encoding for binary features
        - Label encoding for all non-binary categorical features
        
        Args:
            df (pd.DataFrame): DataFrame to process
            categorical_features (list): List of categorical column names
            
        Returns:
            tuple: (encoded_df, encoding_methods_dict)
        """
        encoding_methods = {}
        
        for column in categorical_features:
            # Skip if column doesn't exist or has all NaN values
            if column not in df.columns or df[column].isna().all():
                continue
                
            # Determine number of unique values (excluding NaN)
            unique_values = df[column].dropna().unique()
            unique_count = len(unique_values)
            
            # For binary features (exactly 2 unique values)
            if unique_count == 2:
                # One-hot encode the binary column
                encoder = OneHotEncoder(sparse_output=False, drop='first')
                
                # Reshape data for OneHotEncoder
                encoded_vals = encoder.fit_transform(df[[column]])
                
                # Keep the original column name
                df[column] = encoded_vals
                encoding_methods[column] = "one_hot_encoding"
                
            # For all non-binary features (both low and high cardinality)
            else:
                # Use Label Encoding 
                le = LabelEncoder()
                # Handle potential NaN values properly
                if df[column].isna().any():
                    # Create a mask for NaN values
                    nan_mask = df[column].isna()
                    # Fill NaN with a placeholder
                    df.loc[~nan_mask, column] = le.fit_transform(df.loc[~nan_mask, column].astype(str))
                    # Restore NaN values
                    df.loc[nan_mask, column] = np.nan
                else:
                    df[column] = le.fit_transform(df[column].astype(str))
                
                encoding_methods[column] = "label_encoding"
        
        return df, encoding_methods
    
    def _detect_outliers(self, series):
        """
        Detect if a series has outliers using IQR method.
        
        Args:
            series (pd.Series): Series to analyze
            
        Returns:
            bool: True if outliers detected, False otherwise
        """
        # Drop NaN values for calculation
        series = series.dropna()
        
        # Skip if series is too short
        if len(series) < 10:
            return False
            
        # Calculate IQR
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Check if there are values outside the bounds
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_ratio = outlier_mask.mean()
        
        # Return True if more than 1% of values are outliers
        return outlier_ratio > 0.01