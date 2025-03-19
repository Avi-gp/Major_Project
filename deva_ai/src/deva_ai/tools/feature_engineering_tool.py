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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureEngineeringInput(BaseModel):
    """Input schema for FeatureEngineeringTool that uses file path."""
    preprocessed_file_path: str = Field(..., description="Absolute path to the preprocessed file.")
    preprocessed_file_name: str = Field(..., description="The name of the preprocessed file.")
    target_column: Optional[str] = Field(None, description="Name of the target column for feature importance evaluation. If not provided, user will be prompted.")

class FeatureEngineeringTool(BaseTool):
    name: str = "Feature Engineering Tool"
    description: str = "Transform preprocessed data by applying scaling techniques for numerical data, encoding methods for categorical data, and feature selection"
    args_schema: Type[BaseModel] = FeatureEngineeringInput

    def _run(self, preprocessed_file_path: str, preprocessed_file_name: str, target_column: Optional[str] = None) -> str:
        """
        Process the preprocessed file and perform comprehensive feature engineering.
        
        Args:
            preprocessed_file_path (str): Absolute path to the preprocessed file
            preprocessed_file_name (str): Name of the preprocessed file
            target_column (str, optional): Name of the target column for feature importance evaluation
            
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
            
            # Prompt for target column if not provided
            if target_column is None:
                print(f"Available columns: {', '.join(df.columns)}")
                target_column = df.columns[-1]
                if target_column.lower() == 'none':
                    target_column = None
                elif target_column not in df.columns:
                    print(f"Warning: '{target_column}' is not a valid column name. Proceeding without target column.")
                    target_column = None
            
            # Identify feature types
            numerical_features, categorical_features = self._identify_feature_types(df)
            
            # Execute feature engineering steps
            df, feature_engineering_stats = self._engineer_features(df, numerical_features, categorical_features, target_column)
            
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
                "target_column": target_column,
                "suggested_features_to_drop": feature_engineering_stats["suggested_features_to_drop"],
                "high_correlation_pairs": feature_engineering_stats["high_correlation_pairs"],
                "variance_filtered_features": feature_engineering_stats["variance_filtered_features"],
                "feature_importance": feature_engineering_stats["feature_importance"],
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
    
    def _engineer_features(self, df, numerical_features, categorical_features, target_column=None):
        """
        Apply feature engineering to the DataFrame.
        
        Args:
            df (pd.DataFrame): The DataFrame to engineer
            numerical_features (list): List of numerical feature names
            categorical_features (list): List of categorical feature names
            target_column (str, optional): Name of the target column
            
        Returns:
            tuple: (engineered_df, feature_engineering_stats_dict)
        """
        feature_engineering_stats = {
            "scaling_methods_applied": {},
            "encoding_methods_applied": {},
            "suggested_features_to_drop": [],
            "high_correlation_pairs": [],
            "variance_filtered_features": [],
            "feature_importance": {}
        }
        
        # Store target column original values for later use in feature importance
        y = None
        if target_column and target_column in df.columns:
            y = df[target_column].copy()
            
        # Part 1: Numerical Feature Scaling
        df, scaling_stats = self._scale_numerical_features(df, numerical_features)
        feature_engineering_stats["scaling_methods_applied"] = scaling_stats
        
        # Part 2: Categorical Feature Encoding
        df, encoding_stats = self._encode_categorical_features(df, categorical_features)
        feature_engineering_stats["encoding_methods_applied"] = encoding_stats
        
        # Part 3: Feature Selection
        # Now create temporary feature lists without the target column for feature selection
        temp_numerical_features = numerical_features.copy()
        temp_categorical_features = categorical_features.copy()
        
        # Remove target column from temporary feature lists only for feature selection
        if target_column:
            if target_column in temp_numerical_features:
                temp_numerical_features.remove(target_column)
            if target_column in temp_categorical_features:
                temp_categorical_features.remove(target_column)
        
        # 3.1: Correlation-based feature selection
        high_corr_pairs = self._find_correlated_features(df)
        feature_engineering_stats["high_correlation_pairs"] = high_corr_pairs
        
        # 3.2: Variance threshold filtering
        low_variance_features = self._find_low_variance_features(df)
        feature_engineering_stats["variance_filtered_features"] = low_variance_features
        
        # 3.3: Feature importance evaluation using Random Forest
        feature_importance = self._evaluate_feature_importance(df, y)
        feature_engineering_stats["feature_importance"] = feature_importance
        
        # 3.4: Suggest features to drop based on previous analyses
        suggested_drops = self._suggest_features_to_drop(
            df, 
            high_corr_pairs, 
            low_variance_features, 
            feature_importance
        )
        feature_engineering_stats["suggested_features_to_drop"] = suggested_drops
        
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
                encoding_methods[column] = "one_hot_encoding_binary"
                
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
    
    def _find_correlated_features(self, df, threshold=0.8):
        """
        Find highly correlated feature pairs.
        
        Args:
            df (pd.DataFrame): DataFrame to analyze
            threshold (float): Correlation threshold
            
        Returns:
            list: List of tuples [(feature1, feature2, correlation_value), ...]
        """
        # Only consider numerical columns for correlation analysis
        numeric_df = df.select_dtypes(include=['number'])
        
        # Skip if there are too few numeric columns
        if numeric_df.shape[1] < 2:
            return []
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr().abs()
        
        # Extract upper triangle excluding diagonal
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find feature pairs with correlation above threshold
        high_corr_pairs = []
        for col1 in upper_tri.columns:
            for col2 in upper_tri.index:
                if upper_tri.loc[col2, col1] > threshold:
                    high_corr_pairs.append(
                        (col1, col2, round(float(upper_tri.loc[col2, col1]), 3))
                    )
        
        return high_corr_pairs
    
    def _find_low_variance_features(self, df, threshold=0.01):
        """
        Find features with variance below threshold.
        
        Args:
            df (pd.DataFrame): DataFrame to analyze
            threshold (float): Variance threshold
            
        Returns:
            list: List of feature names with low variance
        """
        # Only consider numerical columns for variance analysis
        numeric_df = df.select_dtypes(include=['number'])
        
        low_variance_features = []
        for column in numeric_df.columns:
            # Calculate variance
            variance = df[column].var()
            
            # Check if variance is below threshold
            if variance < threshold:
                low_variance_features.append({
                    "feature": column,
                    "variance": float(variance)
                })
        
        return low_variance_features
    
    def _evaluate_feature_importance(self, df, y=None):
        """
        Evaluate feature importance using a Random Forest model.
        
        Args:
            df (pd.DataFrame): DataFrame with features
            y (pd.Series, optional): Target variable
            
        Returns:
            dict: Dictionary of feature importance scores
        """
        # Skip if no target provided
        if y is None:
            return {"target_column_used": False, "features": {}}
        
        # Consider all numeric columns for feature importance
        numeric_df = df.select_dtypes(include=['number'])
        
        # Identify and exclude target column if it's in the dataframe
        target_column_name = None
        if y is not None and isinstance(y, pd.Series) and y.name in df.columns:
            target_column_name = y.name
            if target_column_name in numeric_df.columns:
                numeric_df = numeric_df.drop(columns=[target_column_name])
        
        # Skip if there are too few features after target removal
        if numeric_df.shape[1] < 1:
            return {"target_column_used": True, "features": {}}
        
        try:
            # Create X dataset for feature importance
            X = numeric_df.fillna(0)  # Simple imputation for any remaining NaN values
            
            # Determine if classification or regression task
            if pd.api.types.is_numeric_dtype(y):
                # For numeric targets, use a regressor
                unique_count = y.nunique()
                if unique_count <= 10:  # Small number of unique values suggests classification
                    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                else:
                    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            else:
                # For categorical targets, use a classifier
                model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            
            # Train the model
            model.fit(X, y)
            
            # Extract feature importances
            importance_dict = {
                "target_column_used": True,
                "features": {}
            }
            
            for i, column in enumerate(X.columns):
                # Round the importance value to 4 decimal places
                importance_dict["features"][column] = round(float(model.feature_importances_[i]), 4)
            
            # Sort by importance (descending)
            importance_dict["features"] = {k: v for k, v in sorted(
                importance_dict["features"].items(), key=lambda item: item[1], reverse=True
            )}
            
            return importance_dict
            
        except Exception as e:
            logging.warning(f"Could not evaluate feature importance: {str(e)}")
            return {"target_column_used": True, "features": {}}
    
    def _suggest_features_to_drop(self, df, high_corr_pairs, low_variance_features, feature_importance):
        """
        Suggest features to drop based on analysis results.
        
        Args:
            df (pd.DataFrame): DataFrame to analyze
            high_corr_pairs (list): List of highly correlated feature pairs
            low_variance_features (list): List of features with low variance
            feature_importance (dict): Dictionary of feature importance scores with nested structure
            
        Returns:
            list: List of dictionaries with feature names and reasons to drop
        """
        suggested_drops = []
        
        # Extract features dict from the new structure
        features_importance = {}
        if feature_importance and "features" in feature_importance:
            features_importance = feature_importance["features"]
        
        # 1. Suggest dropping one feature from each highly correlated pair
        already_suggested = set()
        
        for feature1, feature2, corr_value in high_corr_pairs:
            # Skip if both features were already suggested
            if feature1 in already_suggested and feature2 in already_suggested:
                continue
                
            # Decide which feature to drop based on feature importance if available
            if feature1 in features_importance and feature2 in features_importance:
                # Drop the less important feature
                if features_importance[feature1] < features_importance[feature2]:
                    feature_to_drop = feature1
                else:
                    feature_to_drop = feature2
            else:
                # If no importance data, suggest the second feature arbitrarily
                feature_to_drop = feature2
            
            # Add to suggestions if not already there
            if feature_to_drop not in already_suggested:
                suggested_drops.append({
                    "feature": feature_to_drop,
                    "reason": f"High correlation ({corr_value}) with {feature1 if feature_to_drop == feature2 else feature2}"
                })
                already_suggested.add(feature_to_drop)
        
        # 2. Suggest dropping features with very low variance
        for feature_info in low_variance_features:
            feature = feature_info["feature"]
            variance = feature_info["variance"]
            
            # Skip if already suggested
            if feature in already_suggested:
                continue
                
            suggested_drops.append({
                "feature": feature,
                "reason": f"Low variance ({variance})"
            })
            already_suggested.add(feature)
        
        # 3. Suggest dropping features with very low importance
        if features_importance:
            # Get features with importance below 1% of max importance
            max_importance = max(features_importance.values()) if features_importance else 0
            importance_threshold = max_importance * 0.01
            
            for feature, importance in features_importance.items():
                # Skip if already suggested
                if feature in already_suggested:
                    continue
                    
                if importance < importance_threshold:
                    suggested_drops.append({
                        "feature": feature,
                        "reason": f"Low feature importance ({importance:.6f})"
                    })
                    already_suggested.add(feature)
        
        return suggested_drops
    
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