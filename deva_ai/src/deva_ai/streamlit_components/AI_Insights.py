import streamlit as st
import pandas as pd
import numpy as np
from helper import helperFunction


def generate_ai_insights(df, target_column, numerical_cols, categorical_cols, target_is_numeric, feature_num_cols):
    """
    Generate comprehensive AI-powered insights about the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset being analyzed
    target_column : str
        Name of the target column for analysis
    numerical_cols : list
        List of numerical column names
    categorical_cols : list
        List of categorical column names
    target_is_numeric : bool
        Whether the target column is numeric
    feature_num_cols : list
        List of numerical features (excluding target if it's numerical)
    
    Returns:
    --------
    str
        Markdown-formatted AI insights
    """
    try:
        # Extract basic statistics for context
        num_rows = len(df)
        num_cols = len(df.columns)
        
        # Check for missing values
        missing_values = df.isnull().sum()
        columns_with_missing = missing_values[missing_values > 0]
        missing_info = ""
        if len(columns_with_missing) > 0:
            missing_info = f"Columns with missing values: {', '.join(columns_with_missing.index.tolist())}"
        
        # Get correlation with target if numeric
        target_correlations = {}
        if target_is_numeric and len(feature_num_cols) > 0:
            target_corr = df[feature_num_cols].corrwith(df[target_column]).abs().sort_values(ascending=False)
            top_correlated = target_corr.head(5)
            target_correlations = dict(zip(top_correlated.index, top_correlated.values))
        
        # Get basic stats for numeric columns
        numeric_stats = df[numerical_cols].describe().transpose()
        
        # Prepare the dataset overview
        dataset_overview = {
            "basic_info": {
                "rows": num_rows,
                "columns": num_cols,
                "numerical_columns": len(numerical_cols),
                "categorical_columns": len(categorical_cols),
                "target_column": target_column,
                "target_type": "numerical" if target_is_numeric else "categorical"
            },
            "missing_data": missing_info,
            "target_correlations": target_correlations,
        }
        
        # Calculate data quality metrics
        data_quality = {
            "completeness": (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
            "duplicates": df.duplicated().sum(),
            "outlier_percentage": {}
        }
        
        # Calculate outlier percentages for numerical columns
        for col in numerical_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)].shape[0]
            data_quality["outlier_percentage"][col] = (outliers / df.shape[0]) * 100
        
        # Calculate segment performance metrics if appropriate categorical columns exist
        segment_metrics = {}
        if target_is_numeric and len(categorical_cols) > 0:
            # Find categorical columns with reasonable number of categories for segmentation
            segment_cols = [col for col in categorical_cols if 2 <= df[col].nunique() <= 10]
            
            for col in segment_cols[:3]:  # Limit to top 3 categorical columns
                segment_metrics[col] = {}
                for segment in df[col].unique():
                    segment_data = df[df[col] == segment]
                    segment_metrics[col][segment] = {
                        "count": len(segment_data),
                        "percentage": (len(segment_data) / len(df)) * 100,
                        "target_mean": segment_data[target_column].mean(),
                        "target_median": segment_data[target_column].median(),
                        "target_std": segment_data[target_column].std()
                    }
        
        # Calculate time-based metrics if datetime columns exist
        time_metrics = {}
        datetime_columns = [col for col in df.columns if pd.api.types.is_datetime64_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col])]
        
        if datetime_columns and target_is_numeric:
            time_col = datetime_columns[0]  # Use the first datetime column
            
            # Convert to datetime if not already
            df[time_col] = pd.to_datetime(df[time_col])
            
            # Group by year-month and calculate target metrics
            df['year_month'] = df[time_col].dt.to_period('M')
            monthly_stats = df.groupby('year_month')[target_column].agg(['mean', 'median', 'std', 'count'])
            
            # Calculate month-over-month changes
            monthly_stats['mom_change'] = monthly_stats['mean'].pct_change() * 100
            
            # Store in time metrics
            time_metrics = {
                "monthly_stats": monthly_stats.tail(6).to_dict(),  # Last 6 months
                "overall_trend": "increasing" if monthly_stats['mean'].iloc[-3:].mean() > monthly_stats['mean'].iloc[-6:-3].mean() else "decreasing"
            }
        
        # Performance comparison metrics
        performance_metrics = {}
        if target_is_numeric:
            performance_metrics = {
                "current_avg": df[target_column].mean(),
                "current_median": df[target_column].median(),
                "best_segment": None,
                "worst_segment": None,
                "potential_uplift": 0
            }
            
            # Find best and worst segments if we have segment data
            if segment_metrics:
                best_segment = {"column": None, "value": None, "performance": float('-inf')}
                worst_segment = {"column": None, "value": None, "performance": float('inf')}
                
                for col, segments in segment_metrics.items():
                    for segment, metrics in segments.items():
                        if metrics["target_mean"] > best_segment["performance"]:
                            best_segment = {"column": col, "value": segment, "performance": metrics["target_mean"]}
                        if metrics["target_mean"] < worst_segment["performance"]:
                            worst_segment = {"column": col, "value": segment, "performance": metrics["target_mean"]}
                
                performance_metrics["best_segment"] = best_segment
                performance_metrics["worst_segment"] = worst_segment
                
                # Calculate potential uplift if worst segment reached average performance
                if worst_segment["performance"] < performance_metrics["current_avg"]:
                    worst_segment_count = segment_metrics[worst_segment["column"]][worst_segment["value"]]["count"]
                    performance_metrics["potential_uplift"] = (performance_metrics["current_avg"] - worst_segment["performance"]) * worst_segment_count
        
        # Feature impact projections (simplified ROI analysis)
        feature_impact = {}
        if target_is_numeric and target_correlations:
            for feature, correlation in target_correlations.items():
                if abs(correlation) > 0.3:  # Consider moderate to strong correlations
                    # Calculate slope of linear relationship (simplified)
                    slope = np.cov(df[feature], df[target_column])[0, 1] / np.var(df[feature])
                    
                    # Project impact of 10% improvement in feature
                    feature_std = df[feature].std()
                    projected_improvement = slope * (feature_std * 0.1)
                    
                    feature_impact[feature] = {
                        "correlation": correlation,
                        "impact_per_std_dev": slope * feature_std,
                        "impact_of_10pct_improvement": projected_improvement,
                        "total_projected_impact": projected_improvement * len(df)
                    }
        
        # Create the analysis prompt
        system_prompt = """
        You are a data science assistant specialized in providing insights about datasets. 
        Your task is to analyze the dataset information provided and generate valuable insights.
        Focus on patterns, relationships, and potential areas of interest within the data.
        Highlight key findings, potential issues, and recommendations based on the data characteristics.
        Structure your analysis in clear sections with markdown formatting, with actionable insights.
        """
        
        analysis_prompt = f"""
        Dataset Overview:
        - Number of rows: {dataset_overview['basic_info']['rows']}
        - Number of columns: {dataset_overview['basic_info']['columns']}
        - Numerical columns: {dataset_overview['basic_info']['numerical_columns']}
        - Categorical columns: {dataset_overview['basic_info']['categorical_columns']}
        - Target column: {dataset_overview['basic_info']['target_column']}
        - Target type: {dataset_overview['basic_info']['target_type']}
        
        Missing data information:
        {dataset_overview['missing_data'] if dataset_overview['missing_data'] else "No missing values in the dataset."}
        
        Top correlated features with target:
        {dataset_overview['target_correlations'] if dataset_overview['target_correlations'] else "Target is categorical or no strong correlations found."}
        
        Data Quality Metrics:
        - Overall completeness: {data_quality.get('completeness', 'N/A')}%
        - Duplicate records: {data_quality.get('duplicates', 'N/A')}
        - Outlier percentages by column: {data_quality.get('outlier_percentage', 'N/A')}
        
        Performance Metrics:
        {performance_metrics if performance_metrics else "No performance metrics available."}
        
        Segment Analysis:
        {segment_metrics if segment_metrics else "No segment analysis available."}
        
        Time-Based Metrics:
        {time_metrics if time_metrics else "No time-based metrics available."}
        
        Feature Impact Projections:
        {feature_impact if feature_impact else "No feature impact projections available."}
        
        Key statistical information:
        {numeric_stats.to_string()}
        
        Based on the actual data visualization results and statistical analysis performed, provide a comprehensive analysis of this dataset with the following sections:
        
        1. **Key Patterns and Relationships**
           - Identify major trends in the data
           - Highlight significant correlations and dependencies
           - Describe distribution patterns of key variables
        
        2. **Potential Issues or Data Quality Concerns**
           - Detail missing data patterns and impact
           - Identify potential outliers and abnormalities
           - Flag inconsistencies or data integrity issues
        
        3. **Suggestions for Further Analysis or Modeling**
           - Recommend specific modeling approaches based on data characteristics
           - Suggest feature engineering opportunities
           - Propose validation strategies appropriate for this dataset
        
        4. **Feature Importance and Target Relationships**
           - Analyze key drivers of the target variable
           - Quantify impact of top features with specific metrics
           - Discuss interaction effects between features
        
        5. **Potential Business Insights**
           - Translate data findings into actionable business recommendations
           - Identify opportunities for process optimization
           - Suggest strategic decisions supported by the data
        
        6. **KPIs and Performance Metrics**
           - Use the calculated performance metrics to establish relevant KPIs
           - Analyze the current performance baselines
           - Suggest reasonable targets and thresholds based on the data
        
        7. **ROI Analysis and Business Impact**
           - Use the feature impact projections to quantify potential ROI
           - Prioritize recommendations based on projected business impact
           - Estimate implementation effort versus return
        
        8. **Segment Analysis and Targeting**
           - Compare performance across the identified segments
           - Highlight opportunities for segment-specific strategies
           - Recommend targeting priorities based on segment metrics
        
        9. **Time-Based Trends and Forecasting**
           - Interpret the time-based metrics and trends
           - Discuss seasonality and cyclical patterns if present
           - Suggest short-term forecasts based on historical patterns
        
        10. **Implementation Roadmap**
            - Suggest prioritized action steps based on all analysis
            - Recommend monitoring metrics for implemented changes
            - Outline potential challenges and mitigation strategies
        
        11. **Key Recommendations**
            - Provide 3-5 clear, prioritized recommendations based on the analysis
            - For each recommendation include:
              * Specific action to take
              * Expected impact (quantified where possible)
              * Implementation complexity (low/medium/high)
              * Timeline for implementation (immediate/short-term/long-term)
              * Success metrics to track
            - Rank recommendations by potential business value
        
        Format your insights with clear headings, bullet points, and where appropriate, estimated quantitative impacts. Focus on being concise yet comprehensive, with emphasis on actionable insights derived directly from the data analysis results.
        """
        
        # Call the helper function to get AI insights
        helper = helperFunction()
        ai_insights = helper.call_llm(system_prompt, analysis_prompt)
        
        return ai_insights
    
    except Exception as e:
        return f"Error generating AI insights: {str(e)}"