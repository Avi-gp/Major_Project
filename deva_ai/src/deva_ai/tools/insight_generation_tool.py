# insight_generation_tool.py
'''from typing import Type, Optional
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import os
import logging
import json
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

class InsightGenerationInput(BaseModel):
    """Input schema for InsightGenerationTool that uses file path."""
    engineered_file_path: str = Field(..., description="Absolute path to the engineered file.")
    engineered_file_name: str = Field(..., description="The name of the engineered file.")
    target_column: str = Field(..., description="The target column for analysis.")

class InsightGenerationTool(BaseTool):
    name: str = "Insight Generation Tool"
    description: str = "Generate insights from engineered data to understand patterns and relationships"
    args_schema: Type[BaseModel] = InsightGenerationInput

    def _run(self, engineered_file_path: str, engineered_file_name: str, target_column: str) -> str:
        """
        Process the engineered file and generate insights.
        
        Args:
            engineered_file_path (str): Absolute path to the engineered file
            engineered_file_name (str): Name of the engineered file
            target_column (str): Name of the target column for analysis
            
        Returns:
            str: JSON string containing insight generation results
        """
        try:
            # Check if file exists
            if not os.path.exists(engineered_file_path):
                raise FileNotFoundError(f"File not found at path: {engineered_file_path}")
            
            # Read file based on extension
            file_extension = os.path.splitext(engineered_file_name)[1].lower()
            
            if file_extension == '.csv':
                df = pd.read_csv(engineered_file_path)
            elif file_extension == '.xlsx':
                df = pd.read_excel(engineered_file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Verify that target column exists
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataset")
            
            # Identify feature types
            numerical_features, categorical_features = self._identify_feature_types(df, target_column)
            
            # Determine if the target is classification or regression
            target_type = self._determine_target_type(df[target_column])
            
            # Calculate missing values percentage
            missing_values_percentage = (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            
            # Create dataset summary
            dataset_summary = {
                "shape": [int(df.shape[0]), int(df.shape[1])],
                "numerical_features": numerical_features,
                "categorical_features": categorical_features,
                "target_variable": target_column,
                "target_type": target_type,
                "missing_values_percentage": float(missing_values_percentage)
            }
            
            # Generate the different analyses
            univariate_analysis = self._generate_univariate_analysis(df, numerical_features, categorical_features)
            bivariate_analysis = self._generate_bivariate_analysis(df, numerical_features, categorical_features, target_column)
            categorical_analysis = self._generate_categorical_analysis(df, numerical_features, categorical_features)
            correlation_analysis = self._generate_correlation_analysis(df, numerical_features + categorical_features)
            feature_importance = self._generate_feature_importance(df, numerical_features + categorical_features, target_column, target_type)
            target_analysis = self._generate_target_analysis(df, numerical_features, categorical_features, target_column)
            
            # Aggregate results according to the task's expected output format
            result = {
                "dataset_summary": dataset_summary,
                "univariate_analysis": univariate_analysis,
                "bivariate_analysis": bivariate_analysis,
                "categorical_analysis": categorical_analysis,
                "correlation_analysis": correlation_analysis,
                "feature_importance": feature_importance,
                "target_analysis": target_analysis,
            }
            
            # Return as a clean JSON string
            return json.dumps(result, default=self._json_serializer)
            
        except Exception as e:
            logging.error(f"Error generating insights: {str(e)}")
            error_result = {
                "error": str(e),
                "message": f"Failed to generate insights for file at {engineered_file_path}"
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
    
    def _identify_feature_types(self, df, target_column):
        """
        Identify numerical and categorical features in the DataFrame.
        
        Args:
            df (pd.DataFrame): The DataFrame to analyze
            target_column (str): The target column to exclude from features
            
        Returns:
            tuple: (numerical_features_list, categorical_features_list)
        """
        numerical_features = []
        categorical_features = []
        
        for column in df.columns:
            # Skip the target column
            if column == target_column:
                continue
                
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
    
    def _determine_target_type(self, target_series):
        """
        Determine if the target column is for classification or regression.
        
        Args:
            target_series (pd.Series): The target column
            
        Returns:
            str: "classification" or "regression"
        """
        # If target is categorical or has few unique values
        if pd.api.types.is_categorical_dtype(target_series) or target_series.dtype == 'object':
            return "classification"
        
        # If numeric but with few unique integer values, likely classification
        unique_count = target_series.nunique()
        if unique_count <= 10 and pd.api.types.is_integer_dtype(target_series):
            return "classification"
        
        # Otherwise assume regression
        return "regression"
    
    def _generate_univariate_analysis(self, df, numerical_features, categorical_features):
        """
        Generate univariate analysis statistics.
        
        Args:
            df (pd.DataFrame): DataFrame to analyze
            numerical_features (list): List of numerical features
            categorical_features (list): List of categorical features
            
        Returns:
            dict: Dictionary containing univariate analysis results
        """
        univariate_analysis = {
            "numerical_features": [],
            "categorical_features": []
        }
        
        # Analyze numerical features
        for feature in numerical_features:
            # Calculate statistics
            stats = df[feature].describe()
            
            # Add to results - format according to the task specification
            univariate_analysis["numerical_features"].append({
                "feature_name": feature,
                "statistics": {
                    "mean": float(stats['mean']),
                    "median": float(df[feature].median()),
                    "std": float(stats['std']),
                    "min": float(stats['min']),
                    "max": float(stats['max']),
                    "q1": float(stats['25%']),
                    "q3": float(stats['75%'])
                }
            })
        
        # Analyze categorical features
        for feature in categorical_features:
            # Get value counts
            value_counts = df[feature].value_counts().sort_values(ascending=False)
            
            # Add to results - format according to the task specification
            univariate_analysis["categorical_features"].append({
                "feature_name": feature,
                "value_counts": value_counts.to_dict(),
                "unique_values": int(df[feature].nunique()),
                "mode": str(df[feature].mode().iloc[0]) if not df[feature].mode().empty else "No mode"
            })
        
        return univariate_analysis
    
    def _generate_bivariate_analysis(self, df, numerical_features, categorical_features, target_column):
        """
        Generate bivariate analysis statistics.
        
        Args:
            df (pd.DataFrame): DataFrame to analyze
            numerical_features (list): List of numerical features
            categorical_features (list): List of categorical features
            target_column (str): Name of the target column
            
        Returns:
            dict: Dictionary containing bivariate analysis results
        """
        bivariate_analysis = {
            "feature_relationships": [],
            "target_relationships": {}
        }
        
        # Create feature relationships
        if len(numerical_features) >= 2:
            # Get correlations with target
            top_features = []
            if target_column in df.columns and pd.api.types.is_numeric_dtype(df[target_column]):
                correlations = df[numerical_features].corrwith(df[target_column]).abs().sort_values(ascending=False)
                top_features = correlations.head(min(5, len(correlations))).index.tolist()
            else:
                # If target is not numeric, just take the first few features
                top_features = numerical_features[:min(5, len(numerical_features))]
            
            # Generate correlations for pairs of top features
            for i in range(len(top_features)):
                for j in range(i+1, len(top_features)):
                    feature1 = top_features[i]
                    feature2 = top_features[j]
                    
                    # Calculate correlation
                    correlation = df[feature1].corr(df[feature2])
                    
                    # Determine relationship type
                    if abs(correlation) > 0.7:
                        relationship_type = "strong linear"
                    elif abs(correlation) > 0.3:
                        relationship_type = "moderate linear"
                    else:
                        relationship_type = "weak or non-linear"
                    
                    # Add to results - format according to the task specification
                    bivariate_analysis["feature_relationships"].append({
                        "features": [feature1, feature2],
                        "correlation": float(correlation),
                        "relationship_type": relationship_type
                    })
        
        # Create target relationships
        for feature in numerical_features:
            # Calculate correlation if both are numeric
            correlation = 0
            if pd.api.types.is_numeric_dtype(df[target_column]):
                correlation = df[feature].corr(df[target_column])
            
            # Add to results - format according to the task specification
            bivariate_analysis["target_relationships"][feature] = {
                "correlation": float(correlation) if pd.api.types.is_numeric_dtype(df[target_column]) else None
            }
        
        return bivariate_analysis
    
    def _generate_categorical_analysis(self, df, numerical_features, categorical_features):
        """
        Generate categorical analysis statistics.
        
        Args:
            df (pd.DataFrame): DataFrame to analyze
            numerical_features (list): List of numerical features
            categorical_features (list): List of categorical features
            
        Returns:
            dict: Dictionary containing categorical analysis results
        """
        categorical_analysis = {
            "category_distributions": {},
            "numerical_by_category": []
        }
        
        # Create category distributions
        for feature in categorical_features:
            # Get value counts
            value_counts = df[feature].value_counts().sort_values(ascending=False)
            
            # Add to results - format according to the task specification
            categorical_analysis["category_distributions"][feature] = {
                "most_common": value_counts.index[0] if not value_counts.empty else "No data",
                "least_common": value_counts.index[-1] if len(value_counts) > 1 else "No data",
                "distribution": value_counts.to_dict()
            }
        
        # Create numerical by category statistics
        for num_feature in numerical_features:
            for cat_feature in categorical_features:
                # Skip if too many categories
                if df[cat_feature].nunique() > 10:
                    continue
                    
                # Calculate group statistics
                group_statistics = {}
                for category in df[cat_feature].unique():
                    subset = df[df[cat_feature] == category][num_feature]
                    if not subset.empty:
                        group_statistics[str(category)] = {
                            "mean": float(subset.mean()),
                            "median": float(subset.median()),
                            "std": float(subset.std()) if len(subset) > 1 else 0.0
                        }
                
                # Add to results - format according to the task specification
                categorical_analysis["numerical_by_category"].append({
                    "numerical": num_feature,
                    "categorical": cat_feature,
                    "group_statistics": group_statistics
                })
        
        return categorical_analysis
    
    def _generate_correlation_analysis(self, df, features):
        """
        Generate correlation analysis statistics.
        
        Args:
            df (pd.DataFrame): DataFrame to analyze
            features (list): List of all features
            
        Returns:
            dict: Dictionary containing correlation analysis results
        """
        correlation_analysis = {
            "high_correlation_pairs": []
        }
        
        # Select only numeric columns
        numeric_df = df[features].select_dtypes(include=['number'])
        
        if numeric_df.shape[1] > 1:  # Need at least 2 numeric columns
            # Calculate correlation matrix
            corr_matrix = numeric_df.corr()
            
            # Find high correlation pairs
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    feature1 = corr_matrix.columns[i]
                    feature2 = corr_matrix.columns[j]
                    correlation = corr_matrix.iloc[i, j]
                    
                    # Add if correlation is high
                    if abs(correlation) > 0.7:
                        correlation_analysis["high_correlation_pairs"].append({
                            "feature1": feature1,
                            "feature2": feature2,
                            "correlation": float(correlation)
                        })
        
        return correlation_analysis
    
    def _generate_feature_importance(self, df, features, target_column, target_type):
        """
        Generate feature importance statistics.
        
        Args:
            df (pd.DataFrame): DataFrame to analyze
            features (list): List of all features
            target_column (str): Name of the target column
            target_type (str): "classification" or "regression"
            
        Returns:
            dict: Dictionary containing feature importance results
        """
        feature_importance = {
            "importance_scores": {},
            "top_features": []
        }
        
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        
        # Make sure target and at least one feature are included
        if target_column in numeric_df.columns and len(numeric_df.columns) > 1:
            # Extract X and y
            X = numeric_df.drop(columns=[target_column])
            y = numeric_df[target_column]
            
            # Choose model based on target type
            if target_type == "classification":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            # Fit model
            try:
                model.fit(X, y)
                
                # Get feature importance
                importance = model.feature_importances_
                
                # Map feature names to importance scores
                importance_dict = dict(zip(X.columns, importance))
                
                # Sort by importance
                sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                
                # Store importance scores
                feature_importance["importance_scores"] = {feature: float(score) for feature, score in importance_dict.items()}
                
                # Store top features
                feature_importance["top_features"] = [feature for feature, _ in sorted_importance[:min(5, len(sorted_importance))]]
                
            except Exception as e:
                logging.error(f"Error in feature importance calculation: {str(e)}")
        
        return feature_importance
    
    def _generate_target_analysis(self, df, numerical_features, categorical_features, target_column):
        """
        Generate target analysis statistics.
        
        Args:
            df (pd.DataFrame): DataFrame to analyze
            numerical_features (list): List of numerical features
            categorical_features (list): List of categorical features
            target_column (str): Name of the target column
            
        Returns:
            dict: Dictionary containing target analysis results
        """
        target_analysis = {
            "target_distribution_statistics": {},
            "key_predictors": [],
            "key_predictor_relationships": {}
        }
        
        # Calculate target distribution statistics if numeric
        if pd.api.types.is_numeric_dtype(df[target_column]):
            stats = df[target_column].describe()
            target_analysis["target_distribution_statistics"] = {
                "mean": float(stats['mean']),
                "median": float(df[target_column].median()),
                "std": float(stats['std']),
                "min": float(stats['min']),
                "max": float(stats['max']),
                "q1": float(stats['25%']),
                "q3": float(stats['75%'])
            }
        else:
            # For categorical target, get distribution
            value_counts = df[target_column].value_counts()
            target_analysis["target_distribution_statistics"] = {
                "class_distribution": value_counts.to_dict(),
                "class_count": int(df[target_column].nunique())
            }
        
        # Try to identify key predictors
        key_predictors = []
        
        # For numeric target, use correlation
        if pd.api.types.is_numeric_dtype(df[target_column]) and numerical_features:
            correlations = df[numerical_features].corrwith(df[target_column]).abs().sort_values(ascending=False)
            key_predictors = correlations.head(min(3, len(correlations))).index.tolist()
        
        # If no or few key predictors, try feature importance
        if len(key_predictors) < 3:
            # Try to get additional key predictors from any feature importance analysis
            numeric_df = df.select_dtypes(include=['number'])
            
            if target_column in numeric_df.columns and len(numeric_df.columns) > 1:
                # Extract X and y
                X = numeric_df.drop(columns=[target_column])
                y = numeric_df[target_column]
                
                # Choose model based on target type
                if df[target_column].nunique() <= 10:
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                
                # Fit model
                try:
                    model.fit(X, y)
                    
                    # Get feature importance
                    importance = model.feature_importances_
                    
                    # Map feature names to importance scores
                    importance_dict = dict(zip(X.columns, importance))
                    
                    # Sort by importance
                    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                    
                    # Add top features to key predictors if not already there
                    for feature, _ in sorted_importance[:min(3, len(sorted_importance))]:
                        if feature not in key_predictors:
                            key_predictors.append(feature)
                            
                            if len(key_predictors) >= 3:
                                break
                                
                except Exception as e:
                    logging.error(f"Error in target analysis feature importance: {str(e)}")
        
        target_analysis["key_predictors"] = key_predictors
        
        # Create relationship information for key predictors
        for feature in key_predictors:
            # Only handle numeric features
            if feature not in df.columns or not pd.api.types.is_numeric_dtype(df[feature]):
                continue
                
            # Calculate correlation if both are numeric
            correlation = 0
            if pd.api.types.is_numeric_dtype(df[target_column]):
                correlation = df[feature].corr(df[target_column])
            
            # Add to results
            target_analysis["key_predictor_relationships"][feature] = {
                "correlation": float(correlation) if pd.api.types.is_numeric_dtype(df[target_column]) else None
            }
        
        return target_analysis
 '''