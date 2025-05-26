import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from helper import helperFunction
import matplotlib.gridspec as gridspec
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import PIL
from streamlit_components.AI_Insights import generate_ai_insights
import re

# Set the maximum image size limit to a higher value
PIL.Image.MAX_IMAGE_PIXELS = None 

def identify_feature_types(df):
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


def display_insight_generation_results(file_path, target_column):
    """
    Main function to generate and display insights based on the provided dataset
    and target column without requiring user selection for each visualization.
    
    Parameters:
    -----------
    file_path : str
        Path to the dataset file (CSV or Excel)
    target_column : str
        Name of the target column for analysis
    """
    try:
        # Load data
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        
        # Set up page layout
        st.markdown('<h2 class="subheader">🔍 Comprehensive Data Insights & Visualizations</h2>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

        # Separate numerical and categorical columns
        numerical_cols, categorical_cols = identify_feature_types(df)
       
        st.markdown("## Column Type Classification")
        st.text("Columns are classified based on thier types and no of unique values.")
        col_info = pd.DataFrame({
            'Data Type': ['Numerical', 'Categorical'],
            'Count': [len(numerical_cols), len(categorical_cols)]
        })
        st.dataframe(col_info.style.set_properties(**{'padding': '10px'}))

        # Display column names in expandable sections for better visibility
        if numerical_cols:
            with st.expander("📊 View all Numerical Columns"):
                # Display numerical columns in a more readable format
                num_cols_df = pd.DataFrame({'Numerical Columns': numerical_cols})
                st.dataframe(num_cols_df, height=min(35 * len(numerical_cols) + 38, 300))

        if categorical_cols:
            with st.expander("📋 View all Categorical Columns"):
                # Display categorical columns in a more readable format
                cat_cols_df = pd.DataFrame({'Categorical Columns': categorical_cols})
                st.dataframe(cat_cols_df, height=min(35 * len(categorical_cols) + 38, 300))
        
        # Exclude target from features if it exists in either list
        if target_column in numerical_cols:
            feature_num_cols = [col for col in numerical_cols if col != target_column]
            target_is_numeric = True
        else:
            feature_num_cols = numerical_cols
            target_is_numeric = False
            
        if target_column in categorical_cols:
            feature_cat_cols = [col for col in categorical_cols if col != target_column]
            target_is_categorical = True
        else:
            feature_cat_cols = categorical_cols
            target_is_categorical = False

        st.markdown("<div style='padding: 20px 0px;'></div>", unsafe_allow_html=True)
        st.markdown("## 📈 Interactive Data Analysis")
        st.markdown("Explore your dataset through different analytical perspectives using the tabs below.")
        st.markdown("<div style='padding: 10px 0px;'></div>", unsafe_allow_html=True)

        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Univariate", "🔄 Bivariate", "📋 Categorical",
        "🔗 Correlation", "🎯 Feature Importance", 
        "🎯 Target Analysis"
        ])
        
        # 1. Univariate Analysis Tab
        with tab1:
            st.markdown('<div', unsafe_allow_html=True)
            st.markdown("### Distribution Of Numerical Features")
            
            # Show top 15 numerical distributions
            if len(feature_num_cols) > 0:
                # Limit to first 15 numerical columns if there are too many
                display_num_cols = feature_num_cols[:15]
                # Calculate optimal grid dimensions
                n_cols = 3
                n_rows = (len(display_num_cols) + n_cols - 1) // n_cols
                
                fig = plt.figure(figsize=(20, 5 * n_rows))
                gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
                
                # Create custom colormap for variety
                colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', 
                         '#99FFCC', '#FFB366', '#FF99FF', '#99FF99', '#FFB366', 
                         '#B366FF', '#FF8533', '#FF99CC', '#99FFCC', '#FFCC99']
                
                for i, col in enumerate(display_num_cols):
                    ax = fig.add_subplot(gs[i//n_cols, i%n_cols])
                    sns.histplot(data=df, x=col, kde=True, ax=ax, 
                               color=colors[i % len(colors)],
                               edgecolor='black', alpha=0.7)
                    ax.set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # Show top 15 categorical distributions
            if len(feature_cat_cols) > 0:
                st.markdown("### Distribution of Categorical Features")
                
                # Limit to first 15 categorical columns
                display_cat_cols = feature_cat_cols[:15]
                
                # Calculate optimal figure height based on number of categories
                total_categories = sum(df[col].nunique() for col in display_cat_cols)
                fig_height = max(4 * len(display_cat_cols), total_categories * 0.4)
                
                fig = plt.figure(figsize=(20, fig_height))
                
                for i, col in enumerate(display_cat_cols):
                    ax = fig.add_subplot(len(display_cat_cols), 1, i+1)
                    
                    # Get value counts and sort them
                    value_counts = df[col].value_counts().sort_index()
                    
                    # Create bar plot with actual values
                    sns.barplot(x=value_counts.index.astype(str), 
                              y=value_counts.values,
                              hue=value_counts.index.astype(str),
                              ax=ax, 
                              palette="viridis",
                              legend=False)
                    
                    # Add percentage labels on top of bars
                    total = value_counts.sum()
                    for j, v in enumerate(value_counts.values):
                        ax.text(j, v, f'{(v/total)*100:.1f}%', 
                               ha='center', va='bottom')
                    
                    ax.set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
                    ax.set_xlabel('Categories')
                    ax.set_ylabel('Count')
                    plt.xticks(rotation=45, ha='right')
                    ax.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # Box plots for numerical features
            if len(feature_num_cols) > 0:
                st.markdown("### Box Plots of Numerical Features")
                
                # Limit to first 15 numerical columns
                display_num_cols = feature_num_cols[:15]
                
                # Calculate optimal grid dimensions
                n_cols = 3
                n_rows = (len(display_num_cols) + n_cols - 1) // n_cols
                
                fig = plt.figure(figsize=(20, 4 * n_rows))
                gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
                
                # Create box plots for each numerical column
                for i, num_col in enumerate(display_num_cols):
                    ax = fig.add_subplot(gs[i//n_cols, i%n_cols])
                    sns.boxplot(data=df, y=num_col, ax=ax, color=colors[i % len(colors)], width=0.5)
                    ax.set_title(f'Box Plot of {num_col}', fontsize=12, fontweight='bold')
                    ax.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # Box plots for categorical columns with user selection (limit of 5 categorical columns)
            if len(feature_cat_cols) > 0:
                st.markdown("### Box Plots with Categorical Features")
                
                # Filter categorical columns with reasonable number of categories to avoid overcrowding
                filtered_cat_cols = [col for col in feature_cat_cols if df[col].nunique() <= 10]
                
                if len(filtered_cat_cols) > 0 and len(feature_num_cols) > 0:
                    # Let user select categorical columns (limit to 5)
                    selected_cat_cols = st.multiselect(
                        "Select categorical columns (max 5):",
                        options=filtered_cat_cols,
                        default=filtered_cat_cols[:1] if filtered_cat_cols else [],
                        max_selections=5
                    )
                    
                    # Let user select numerical columns (limit to 5)
                    if target_is_numeric and len(feature_num_cols) > 0:
                        # Get correlations to show most relevant numerical columns by default
                        corr_with_target = df[feature_num_cols].corrwith(df[target_column]).abs().sort_values(ascending=False)
                        default_num_cols = corr_with_target.head(2).index.tolist()
                    else:
                        default_num_cols = feature_num_cols[:2] if len(feature_num_cols) >= 2 else feature_num_cols
                    
                    selected_num_cols = st.multiselect(
                        "Select numerical columns (max 5):",
                        options=feature_num_cols,
                        default=default_num_cols,
                        max_selections=5
                    )
                    
                    # Ensure we don't exceed total of 15 box plots
                    total_plots = len(selected_num_cols) * len(selected_cat_cols)
                    if total_plots > 15:
                        st.warning(f"Selected combinations would create {total_plots} plots. Limiting to 15 plots.")
                        # Determine how many numerical columns to use
                        max_num_cols = min(len(selected_num_cols), 15 // max(1, len(selected_cat_cols)))
                        selected_num_cols = selected_num_cols[:max_num_cols]
                    
                    if selected_num_cols and selected_cat_cols:
                        for num_col in selected_num_cols:
                            fig = plt.figure(figsize=(18, 5 * len(selected_cat_cols)))
                            
                            for i, cat_col in enumerate(selected_cat_cols):
                                ax = fig.add_subplot(len(selected_cat_cols), 1, i+1)
                                
                                # Create box plot with categorical column on x-axis and numerical column on y-axis
                                sns.boxplot(data=df, x=cat_col, y=num_col, hue=cat_col, ax=ax, palette='viridis', legend=False)
                                
                                ax.set_title(f'Box Plot of {num_col} by {cat_col}', fontsize=14, fontweight='bold')
                                ax.set_xlabel(cat_col, fontsize=12)
                                ax.set_ylabel(num_col, fontsize=12)
                                plt.xticks(rotation=45, ha='right')
                                ax.grid(True, alpha=0.3, axis='y')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                    elif not selected_cat_cols:
                        st.info("Please select at least one categorical column.")
                    elif not selected_num_cols:
                        st.info("Please select at least one numerical column.")
                elif len(filtered_cat_cols) == 0:
                    st.info("No suitable categorical columns found for box plots (limited to categories with 10 or fewer unique values).")
                elif len(feature_num_cols) == 0:
                    st.info("No numerical columns available for creating box plots with categorical features.")
            else:
                st.info("No categorical columns found in the dataset.")


        # 2. Bivariate Analysis Tab
        with tab2:
            st.markdown('<div>', unsafe_allow_html=True)
            
            # Time Series Analysis (New Section)
            # Check if any datetime columns exist
            datetime_columns = [col for col in df.columns if pd.api.types.is_datetime64_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col])]
            
            if datetime_columns and len(feature_num_cols) > 0:
                st.markdown("### Time Series Analysis")
                
                # Let user select time column and numeric columns for analysis
                time_col = st.selectbox("Select time column for analysis:", datetime_columns)
                
                # Multi-select for numerical columns (limit to top 5 for readability)
                if len(feature_num_cols) > 5:
                    default_selections = feature_num_cols[:3]
                else:
                    default_selections = feature_num_cols[:2] if len(feature_num_cols) >= 2 else feature_num_cols
                
                selected_num_cols = st.multiselect(
                    "Select numerical variables to plot over time (max 5):",
                    options=feature_num_cols,
                    default=default_selections
                )
                
                if time_col and selected_num_cols:
                    # Limit to 5 columns for readability
                    plot_cols = selected_num_cols[:5]
                    
                    # Sort the dataframe by time column
                    df_time = df.sort_values(by=time_col)
                    
                    # Create plotly figure for interactive time series
                    fig = make_subplots(rows=len(plot_cols), cols=1, 
                                    shared_xaxes=True,
                                    vertical_spacing=0.1,
                                    subplot_titles=[f"{col} over Time" for col in plot_cols])
                    
                    for i, col in enumerate(plot_cols):
                        fig.add_trace(
                            go.Scatter(
                                x=df_time[time_col], 
                                y=df_time[col],
                                mode='lines+markers',
                                name=col,
                                line=dict(width=2),
                                marker=dict(size=6)
                            ),
                            row=i+1, col=1
                        )
                        
                        # Add trend line (rolling average)
                        try:
                            window_size = max(7, len(df_time) // 20)  # Adaptive window size
                            rolling_avg = df_time[col].rolling(window=window_size).mean()
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=df_time[time_col],
                                    y=rolling_avg,
                                    mode='lines',
                                    name=f"{col} Trend",
                                    line=dict(width=3, dash='dash', color='rgba(255, 165, 0, 0.8)')
                                ),
                                row=i+1, col=1
                            )
                        except Exception as e:
                            st.warning(f"Could not add trend line for {col}: {str(e)}")
                    
                    # Update layout
                    fig.update_layout(
                        height=300 * len(plot_cols),
                        title_text="Time Series Analysis",
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    # Update all xaxes properties
                    fig.update_xaxes(title_text="Time", showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.5)')
                    
                    # Update all yaxes properties
                    for i, col in enumerate(plot_cols):
                        fig.update_yaxes(title_text=col, row=i+1, col=1, 
                                        showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.5)')
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Modified approach with user selection
            if len(feature_num_cols) >= 2:
                st.markdown("### Relationships Between Numerical Features")
                
                # Select default features based on correlation with target
                if target_is_numeric:
                    correlations = abs(df[feature_num_cols].corrwith(df[target_column]))
                    default_features = correlations.nlargest(min(5, len(correlations))).index.tolist()
                else:
                    default_features = feature_num_cols[:min(5, len(feature_num_cols))]
                
                # Allow user to select features (with defaults pre-selected)
                selected_features = st.multiselect(
                    "Select numerical features for scatter plot analysis (2-6 recommended):",
                    options=feature_num_cols,
                    default=default_features[:min(3, len(default_features))]
                )
                
                # Create scatter plots only if 2+ features are selected
                if len(selected_features) >= 2:
                    # Create scatter plots for selected numerical pairs
                    pairs = [(i, j) for i in range(len(selected_features)) 
                            for j in range(i+1, len(selected_features))]
                    
                    # Show warning if too many plots will be generated
                    if len(pairs) > 15:
                        st.warning(f"Your selection will generate {len(pairs)} scatter plots. Consider selecting fewer features for better visualization.")
                    
                    n_cols = 2
                    n_rows = (len(pairs) + n_cols - 1) // n_cols
                    
                    fig = plt.figure(figsize=(20, 6 * n_rows))
                    
                    # Custom color palette for scatter plots
                    scatter_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD']
                    
                    for idx, (i, j) in enumerate(pairs):
                        ax = fig.add_subplot(n_rows, n_cols, idx + 1)
                        col1, col2 = selected_features[i], selected_features[j]
                        
                        # Calculate correlation
                        corr = df[col1].corr(df[col2])
                        
                        # Create scatter plot with custom colors
                        sns.scatterplot(data=df, x=col1, y=col2,
                                    color=scatter_colors[idx % len(scatter_colors)],
                                    alpha=0.6, ax=ax)
                        
                        ax.set_title(f'{col1} vs {col2}\n(r={corr:.2f})', 
                                fontsize=12, fontweight='bold')
                        ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                elif len(selected_features) == 1:
                    st.info("Please select at least one more feature to create scatter plots.")
                else:
                    st.info("Please select features to visualize relationships.")

        # 3. Categorical Analysis Tab
        with tab3:
            st.markdown('<div>', unsafe_allow_html=True)
            if len(feature_cat_cols) > 0:
                st.markdown("### Categorical Analysis")
                
                # Display categorical variables summary
                categorical_summary = pd.DataFrame({
                    'Variable': feature_cat_cols,
                    'Unique Values': [df[col].nunique() for col in feature_cat_cols],
                    'Most Common': [df[col].mode()[0] if not df[col].mode().empty else "N/A" for col in feature_cat_cols],
                    'Missing Values': [df[col].isnull().sum() for col in feature_cat_cols]
                })
                st.dataframe(categorical_summary)
                
                # Automatically display visualizations for up to 15 categorical columns
                display_cat_cols = feature_cat_cols[:min(15, len(feature_cat_cols))]
                
                if display_cat_cols:
                    st.markdown("### Categorical Distributions")
                    
                    n_cols = 3
                    
                    for i in range(0, len(display_cat_cols), n_cols):
                        cols = st.columns(n_cols)
                        for j in range(n_cols):
                            if i + j < len(display_cat_cols):
                                cat_col = display_cat_cols[i + j]
                                with cols[j]:
                                    # Value counts for this categorical column
                                    value_counts = df[cat_col].value_counts()
                                    
                                    # Create a pie chart for distribution
                                    fig = px.pie(
                                        values=value_counts.values,
                                        names=value_counts.index,
                                        title=f'Distribution of {cat_col}'
                                    )
                                    fig.update_layout(height=350, margin=dict(t=50, b=20, l=20, r=20))
                                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Categorical relationships - modified to allow selection of both primary and secondary columns
                    st.markdown("### Categorical Relationships")

                    # Filter columns with 15 or fewer unique values
                    cat_cols_for_relationship = [col for col in display_cat_cols if df[col].nunique() <= 15]

                    if len(cat_cols_for_relationship) >= 2:
                        # Initialize session state for relationship analysis
                        if 'relationship_analysis_state' not in st.session_state:
                            st.session_state.relationship_analysis_state = {
                                'primary_col': cat_cols_for_relationship[0],
                                'secondary_col': cat_cols_for_relationship[1] if len(cat_cols_for_relationship) > 1 else None
                            }

                        # Create two columns for the selection widgets
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Create selectbox for the primary column
                            selected_primary_col = st.selectbox(
                                "Select primary categorical column:", 
                                options=cat_cols_for_relationship,
                                key="primary_cat_col_selectbox",
                                index=cat_cols_for_relationship.index(st.session_state.relationship_analysis_state['primary_col']) 
                                    if st.session_state.relationship_analysis_state['primary_col'] in cat_cols_for_relationship else 0
                            )
                        
                        # Update secondary column options (excluding the primary column)
                        secondary_options = [col for col in cat_cols_for_relationship if col != selected_primary_col]
                        
                        with col2:
                            # Create selectbox for the secondary column
                            selected_secondary_col = st.selectbox(
                                "Select secondary categorical column:",
                                options=secondary_options,
                                key="secondary_cat_col_selectbox",
                                index=secondary_options.index(st.session_state.relationship_analysis_state['secondary_col']) 
                                    if st.session_state.relationship_analysis_state['secondary_col'] in secondary_options else 0
                            )

                        # Update session state
                        st.session_state.relationship_analysis_state['primary_col'] = selected_primary_col
                        st.session_state.relationship_analysis_state['secondary_col'] = selected_secondary_col

                        if selected_primary_col and selected_secondary_col:
                            st.subheader(f"Relationship: {selected_primary_col} vs {selected_secondary_col}")
                            
                            # Create contingency table for stacked bar chart
                            contingency_table = pd.crosstab(
                                df[selected_primary_col], 
                                df[selected_secondary_col], 
                                normalize='index'
                            )
                            
                            # Create stacked bar chart
                            fig = px.bar(
                                contingency_table, 
                                title=f'Relationship between {selected_primary_col} and {selected_secondary_col}',
                                barmode='stack',
                                height=500
                            )
                            
                            # Improve layout
                            fig.update_layout(
                                xaxis_title=selected_primary_col,
                                yaxis_title="Proportion",
                                legend_title=selected_secondary_col
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Also add a heatmap view of the relationship
                            st.subheader(f"Heatmap: {selected_primary_col} vs {selected_secondary_col}")
                            
                            # Create count contingency table (not normalized)
                            count_contingency = pd.crosstab(
                                df[selected_primary_col], 
                                df[selected_secondary_col]
                            )
                            
                            # Create heatmap
                            fig = px.imshow(
                                count_contingency,
                                labels=dict(x=selected_secondary_col, y=selected_primary_col, color="Count"),
                                x=count_contingency.columns,
                                y=count_contingency.index,
                                color_continuous_scale="Viridis",
                                aspect="auto"
                            )
                            
                            # Add count values as text annotations
                            annotations = []
                            for i, row in enumerate(count_contingency.index):
                                for j, col in enumerate(count_contingency.columns):
                                    annotations.append(
                                        dict(
                                            x=j,
                                            y=i,
                                            text=str(count_contingency.iloc[i, j]),
                                            font=dict(color='white' if count_contingency.iloc[i, j] > count_contingency.values.mean() else 'black'),
                                            showarrow=False
                                        )
                                    )
                            fig.update_layout(annotations=annotations)
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add option to view multiple relationships
                            st.markdown("### Additional Relationships")
                            show_more = st.checkbox("Show relationships with other categorical variables")
                            
                            if show_more:
                                # Let user select additional secondary columns
                                additional_secondary_cols = st.multiselect(
                                    f"Select additional categorical variables to compare with {selected_primary_col}:",
                                    [col for col in secondary_options if col != selected_secondary_col],
                                    default=[]
                                )
                                
                                for secondary_col in additional_secondary_cols:
                                    st.subheader(f"Relationship: {selected_primary_col} vs {secondary_col}")
                                    
                                    # Create contingency table for stacked bar chart
                                    contingency_table = pd.crosstab(
                                        df[selected_primary_col], 
                                        df[secondary_col], 
                                        normalize='index'
                                    )

                                    # Create stacked bar chart
                                    fig = px.bar(
                                        contingency_table, 
                                        title=f'Relationship between {selected_primary_col} and {secondary_col}',
                                        barmode='stack',
                                        height=500
                                    )
                                    
                                    # Improve layout
                                    fig.update_layout(
                                        xaxis_title=selected_primary_col,
                                        yaxis_title="Proportion",
                                        legend_title=secondary_col
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Add heatmap
                                    count_contingency = pd.crosstab(
                                        df[selected_primary_col], 
                                        df[secondary_col]
                                    )
                                    
                                    fig = px.imshow(
                                        count_contingency,
                                        labels=dict(x=secondary_col, y=selected_primary_col, color="Count"),
                                        x=count_contingency.columns,
                                        y=count_contingency.index,
                                        color_continuous_scale="Viridis",
                                        aspect="auto"
                                    )
                                    
                                    # Add count values as text annotations
                                    annotations = []
                                    for i, row in enumerate(count_contingency.index):
                                        for j, col in enumerate(count_contingency.columns):
                                            annotations.append(
                                                dict(
                                                    x=j,
                                                    y=i,
                                                    text=str(count_contingency.iloc[i, j]),
                                                    font=dict(color='white' if count_contingency.iloc[i, j] > count_contingency.values.mean() else 'black'),
                                                    showarrow=False
                                                )
                                            )
                                    fig.update_layout(annotations=annotations)
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Need at least 2 categorical columns with 15 or fewer unique values to analyze relationships.")

        # 4. Correlation Analysis Tab
        with tab4:
            st.markdown('<div>', unsafe_allow_html=True)
            if len(numerical_cols) > 1:
                st.markdown("### Correlation Matrix")
                
                # Calculate correlation for numerical columns
                corr = df[numerical_cols].corr()
                
                # Create interactive correlation heatmap using plotly
                fig = px.imshow(
                    corr,
                    labels=dict(color="Correlation"),
                    x=corr.columns,
                    y=corr.columns,
                    color_continuous_scale="RdBu",
                    aspect="auto"
                )
                
                # Update layout for better visualization
                fig.update_layout(
                    title="Correlation Matrix",
                    title_x=0.5,
                    width=800,
                    height=800,
                    xaxis_tickangle=-45
                )
                
                # Add correlation values as text annotations
                annotations = []
                for i in range(len(corr.columns)):
                    for j in range(len(corr.columns)):
                        annotations.append(
                            dict(
                                x=i,
                                y=j,
                                text=f"{corr.iloc[i, j]:.2f}",
                                font=dict(color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black'),
                                showarrow=False
                            )
                        )
                fig.update_layout(annotations=annotations)
                
                # Display the interactive heatmap
                st.plotly_chart(fig)
                
                # Add correlation strength analysis
                st.markdown("### Strong Correlations Analysis")
                
                # Get strong correlations (absolute value > 0.5)
                strong_corr = []
                for i in range(len(corr.columns)):
                    for j in range(i+1, len(corr.columns)):
                        if abs(corr.iloc[i, j]) > 0.5:
                            strong_corr.append({
                                'Variables': f"{corr.columns[i]} vs {corr.columns[j]}",
                                'Correlation': corr.iloc[i, j]
                            })
                
                if strong_corr:
                    strong_corr_df = pd.DataFrame(strong_corr)
                    strong_corr_df = strong_corr_df.sort_values('Correlation', key=abs, ascending=False)
                    
                    # Display strong correlations with color coding
                    def color_corr(val):
                        color = 'red' if val < -0.5 else 'green'
                        return f'color: {color}'
                    
                    st.dataframe(strong_corr_df.style.map(color_corr, subset=['Correlation']))
                else:
                    st.info("No strong correlations (|r| > 0.5) found between numerical variables.")
            
            else:
                st.info("Not enough numerical columns for correlation analysis.")
                
            st.markdown('</div>', unsafe_allow_html=True)

        # 5. Feature Importance Analysis Tab
        with tab5:
            st.markdown('<div>', unsafe_allow_html=True)
            st.markdown("### Feature Importance Analysis")
            
            try:
                # Prepare features and target
                X = df.drop(columns=[target_column])
                
                # Handle categorical features
                X_encoded = pd.get_dummies(X, drop_first=True)
                
                # Determine if it's a classification or regression problem
                is_classification = df[target_column].dtype == 'object' or len(df[target_column].unique()) < 10
                
                if is_classification:
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    y = pd.Categorical(df[target_column]).codes
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    y = df[target_column]
                
                # Fit model
                with st.spinner("Training model for feature importance..."):
                    model.fit(X_encoded, y)
                
                # Get feature importance
                feature_importance = pd.DataFrame({
                    'Feature': X_encoded.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                # Display top 15 important features
                top_features = feature_importance.head(15)

                # Create interactive bar chart for feature importance using Plotly
                fig = px.bar(
                    top_features,
                    y='Feature',
                    x='Importance',
                    orientation='h',
                    title='Feature Importance from Random Forest',
                    color='Importance',
                    color_continuous_scale='Spectral',
                    labels={'Importance': 'Importance Score', 'Feature': 'Feature Name'},
                    height=600
                )

                # Improve layout
                fig.update_layout(
                    title_x=0.5,
                    xaxis_title='Importance Score',
                    yaxis_title='Features',
                    yaxis={'categoryorder': 'total ascending'},
                    coloraxis_showscale=True,
                    margin=dict(l=20, r=20, t=60, b=20)
                )

                # Add precise importance values as text
                fig.update_traces(
                    texttemplate='%{x:.3f}',
                    textposition='outside'
                )

                st.plotly_chart(fig, use_container_width=True)

                # Add feature importance explanation
                st.markdown("""
                ### Understanding Feature Importance
                - **What it shows:** The chart above displays the relative importance of each feature in predicting the target variable.
                - **How to interpret:** Larger values indicate features that have a stronger influence on the prediction.
                - **Note:** Feature importance doesn't indicate the direction of impact (positive or negative), only magnitude.
                """)
                
            except Exception as e:
                st.error(f"Error in feature importance analysis: {str(e)}")
                
            st.markdown('</div>', unsafe_allow_html=True)

        # 6. Target-Based Analysis Tab
        with tab6:
            st.markdown('<div>', unsafe_allow_html=True)
            st.markdown(f"### Analysis of Target Variable: {target_column}")
            
            if target_is_numeric:
                # Distribution of numeric target
                fig = plt.figure(figsize=(18, 10))
                
                # Subplot 1: Histogram with KDE
                plt.subplot(2, 2, 1)
                sns.histplot(data=df, x=target_column, kde=True, color='navy', alpha=0.7)
                plt.title(f'Distribution of {target_column}', fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3)
                
                # Subplot 2: Box plot
                plt.subplot(2, 2, 2)
                sns.boxplot(data=df, y=target_column, color='midnightblue')
                plt.title(f'Box Plot of {target_column}', fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3)
                
                # Subplot 3: QQ plot to check normality
                plt.subplot(2, 2, 3)
                from scipy import stats
                stats.probplot(df[target_column].dropna(), plot=plt)
                plt.title(f'QQ Plot of {target_column}', fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3)
                
                # Subplot 4: ECDF
                plt.subplot(2, 2, 4)
                sns.ecdfplot(data=df, x=target_column, color='darkblue')
                plt.title(f'ECDF of {target_column}', fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Relationship with key predictors
                st.markdown(f"### {target_column} Relationship with Key Predictors")
                
                # Get top numerical predictors based on correlation
                if len(feature_num_cols) > 0:
                    target_corr = df[feature_num_cols].corrwith(df[target_column]).abs().sort_values(ascending=False)
                    top_num_predictors = target_corr.index[:min(4, len(target_corr))].tolist()
                    
                    # Create scatter plots with regression line
                    if len(top_num_predictors) > 0:
                        fig = plt.figure(figsize=(18, 4 * len(top_num_predictors)))
                        
                        for i, col in enumerate(top_num_predictors):
                            plt.subplot(len(top_num_predictors), 1, i+1)
                            sns.regplot(data=df, x=col, y=target_column, 
                                      scatter_kws={'alpha': 0.5, 'color': 'darkblue'}, 
                                      line_kws={'color': 'red'})
                            
                            # Calculate correlation coefficient
                            corr = df[col].corr(df[target_column])
                            plt.title(f'{target_column} vs {col} (r={corr:.3f})', fontsize=14, fontweight='bold')
                            plt.grid(True, alpha=0.3)
                            
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
            
            else:
                # Distribution of categorical target
                fig = plt.figure(figsize=(14, 8))
                
                # Plot 1: Count plot
                plt.subplot(1, 2, 1)
                value_counts = df[target_column].value_counts().sort_values(ascending=False)

                # Use a different color palette with hue parameter
                bars = sns.barplot(
                    x=value_counts.index, 
                    y=value_counts.values, 
                    hue=value_counts.index,  # Add hue parameter
                    palette='viridis',
                    legend=False  # Disable legend
                )

                # Add percentage labels
                total = sum(value_counts)
                for i, v in enumerate(value_counts.values):
                    plt.text(i, v + 5, f"{v/total*100:.1f}%", ha='center')
                    
                plt.title(f'Distribution of {target_column}', fontsize=14, fontweight='bold')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3, axis='y')
                
                # Plot 2: Pie chart
                plt.subplot(1, 2, 2)
                plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', 
                      startangle=90, colors=sns.color_palette('viridis', len(value_counts)))
                plt.axis('equal')
                plt.title(f'Proportion of {target_column} Classes', fontsize=14, fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Create mosaic plots for categorical features
                if len(feature_cat_cols) > 0:
                    st.markdown(f"### {target_column} Distribution Across Categories")
                    
                    # Select top categorical columns
                    top_cat_cols = [col for col in feature_cat_cols if df[col].nunique() < 10][:3]
                    
                    if len(top_cat_cols) > 0:
                        fig = plt.figure(figsize=(18, 6 * len(top_cat_cols)))
                        
                        for i, col in enumerate(top_cat_cols):
                            plt.subplot(len(top_cat_cols), 1, i+1)
                            
                            # Create a crosstab
                            cross_tab = pd.crosstab(df[col], df[target_column], normalize='index')
                            
                            # Plot stacked bar
                            cross_tab.plot(kind='bar', stacked=True, colormap='viridis', ax=plt.gca())
                            plt.title(f'Distribution of {target_column} by {col}', fontsize=14, fontweight='bold')
                            plt.xlabel(col)
                            plt.ylabel('Proportion')
                            plt.legend(title=target_column)
                            plt.grid(True, alpha=0.3, axis='y')
                            
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
            
            st.markdown('</div>', unsafe_allow_html=True)

        # AI-Insights Section    
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("## 🤖 AI-Powered Insights")
        st.markdown("<hr>", unsafe_allow_html=True)

        # Generate AI insights using the new function
        with st.spinner("Generating AI insights..."):
            ai_insights = generate_ai_insights(
                df, 
                target_column, 
                numerical_cols, 
                categorical_cols, 
                target_is_numeric, 
                feature_num_cols
            )
            
            if ai_insights and not ai_insights.startswith("Error"):
                # Split the insights into sections by headings
                insight_sections = []
                current_section = ""
                current_title = "Key Insights"
                
                for line in ai_insights.split('\n'):
                    if line.startswith('###'):
                        if current_section:
                            insight_sections.append((current_title, current_section))
                            current_section = ""
                        current_title = line.replace('###', '').strip()
                    else:
                        current_section += line + '\n'
                
                # Add the last section
                if current_section:
                    insight_sections.append((current_title, current_section))
                
                # Display insights in formatted sections
                for title, content in insight_sections:
                    st.markdown(f"### {title}")
                    st.markdown(content)
                    # Add a subtle divider between sections
                    st.markdown("<hr style='border: 0; height: 1px; background-color: #e0e0e0; margin: 20px 0;'>", unsafe_allow_html=True)
                
                def format_markdown_for_download(ai_insights, target_column):
                    """
                    Format the AI insights as a clean markdown file with improved table display
                    
                    Parameters:
                    -----------
                    ai_insights : str
                        The AI-generated insights text
                    target_column : str
                        The name of the target variable
                    
                    Returns:
                    --------
                    str
                        Properly formatted markdown content
                    """
                    # Start with the title and target info
                    md_content = f"AI-Powered Insights for Dataset Analysis\n================================================\n\n**Target Variable: {target_column}**\n\n"
                    
                    # Clean up and standardize the insights
                    cleaned_insights = ai_insights
                    
                    # Replace markdown headers with custom styled headers (no # visible)
                    cleaned_insights = re.sub(r'^##\s+(.+)$', r'\1\n' + '=' * 40, cleaned_insights, flags=re.MULTILINE)
                    cleaned_insights = re.sub(r'^###\s+(.+)$', r'\1\n' + '-' * 30, cleaned_insights, flags=re.MULTILINE)
                    
                    # Ensure bullet points use hyphens consistently
                    cleaned_insights = re.sub(r'^•\s+(.+)$', r'- \1', cleaned_insights, flags=re.MULTILINE)
                    
                    # Improve table formatting - especially for the business value table
                    def format_table_row(row_text):
                        cells = row_text.split('|')
                        return '| ' + ' | '.join(c.strip() for c in cells if c.strip()) + ' |'
                    
                    # Find table-like sections and format them better
                    lines = cleaned_insights.split('\n')
                    for i in range(len(lines)):
                        if '|' in lines[i] and '---' in lines[i]:
                            # This looks like a table header row, format it and surrounding rows
                            header_idx = i
                            # Format the header row
                            lines[header_idx] = format_table_row(lines[header_idx])
                            
                            # Format divider row
                            if header_idx + 1 < len(lines) and '|' in lines[header_idx + 1] and '---' in lines[header_idx + 1]:
                                divider_cells = lines[header_idx + 1].split('|')
                                formatted_divider = '|'
                                for cell in divider_cells:
                                    if cell.strip():
                                        formatted_divider += ' ' + '-' * 10 + ' |'
                                lines[header_idx + 1] = formatted_divider
                            
                            # Format data rows
                            row_idx = header_idx + 2
                            while row_idx < len(lines) and '|' in lines[row_idx]:
                                lines[row_idx] = format_table_row(lines[row_idx])
                                row_idx += 1
                    
                    cleaned_insights = '\n'.join(lines)
                    
                    # Add proper spacing between sections
                    cleaned_insights = re.sub(r'(.+)\n(=+|-+)', r'\1\n\2\n\n', cleaned_insights)
                    
                    md_content += cleaned_insights
                    
                    return md_content
                
                # Process insights for better formatting
                md_content = format_markdown_for_download(ai_insights, target_column)

                # Download button with updated styling
                st.download_button(
                    label="📝 Download Insight Report",
                    data=md_content,
                    file_name=f"ai_insights_{target_column}.md",
                    mime="text/markdown",
                    help="Download the AI-generated insights as a markdown file."
                )
            else:
                st.error("Unable to generate AI insights. Please try again later.")
                if ai_insights and ai_insights.startswith("Error"):
                    st.error(ai_insights)
                                 
    except Exception as e:
        st.error(f"Error: {str(e)}")

