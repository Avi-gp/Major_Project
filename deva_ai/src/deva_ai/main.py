import json
import re
import pandas as pd
import streamlit as st
import os
from crew import DevaAi

# Set the page configuration as the first Streamlit command
st.set_page_config(page_title="DEVA AI : AI-Powered Data Analytics", layout="wide")

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

def preprocess_for_json(raw_content):
    """Preprocess raw content to ensure it is valid JSON"""
    # Replace tuples with lists (tuples use round brackets)
    raw_content = re.sub(r'\((.*?)\)', r'[\1]', raw_content)
    
    # Additional fixes for any other non-JSON-friendly content can be added here
    # For example: replace `...` or other placeholders with actual content or remove them.
    raw_content = raw_content.replace('...', '')
    
    return raw_content

def main():
    # Custom CSS for styling
    st.markdown("""
        <style>
            .title {
                font-family: "Times New Roman", Times, serif;
                text-align: center;
                font-size: 30px;
                color: #4CAF50;
                padding: 10px 0px 10px 0px;
            }
            .subheader {
                font-family: "Times New Roman", Times, serif;
                font-size: 24px;
            }
            .header {
                font-family: "Times New Roman", Times, serif;
                font-size: 20px;
            }
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                font-family: "Times New Roman", Times, serif;
            }
        </style>
    """, unsafe_allow_html=True)

    # Title with icon
    st.markdown('<h1 class="title">🧑‍💼 DEVA AI : AI-Powered Data Analytics </h1>', unsafe_allow_html=True)
    st.write("Upload your dataset and get AI-powered insights and analysis.")
    
    uploaded_file = st.file_uploader("Upload your dataset file", type=["csv", "xlsx"])
    
    if uploaded_file:
        try:
            # Save file and get absolute path
            file_path = save_uploadedfile(uploaded_file)
            
            if file_path:
                st.success(f"File successfully saved at: {file_path}")
                
                # Create DevaAi instance
                deva = DevaAi()
                
                # Prepare input for CrewAI with absolute file path
                inputs = {
                    "file_path": file_path,
                    "file_name": uploaded_file.name
                }
                
                # Process with CrewAI
                with st.spinner('Processing dataset...'):
                    crew_output = deva.crew().kickoff(inputs=inputs)
                
                # Preprocess the raw content to ensure it's valid JSON
                try:
                    raw_content = crew_output.raw.strip("`")  # Remove extra backticks if present
                    raw_content = preprocess_for_json(raw_content)  # Preprocess to valid JSON
                    
                    # Attempt to parse as JSON
                    result = json.loads(raw_content)
                    
                    # Display parsed output
                    if "dataset_shape" in result:
                        st.subheader("Dataset Shape")
                        st.write(f"Dataset shape: {result['dataset_shape']}")

                    if "preview" in result:
                        st.subheader("Data Preview")
                        preview_df = pd.DataFrame(result["preview"])
                        st.dataframe(preview_df)

                    if "numeric_stats" in result:
                        st.subheader("Numeric Column Statistics")
                        stats_df = pd.DataFrame(result["numeric_stats"]).T  
                        st.dataframe(stats_df)
                        
                except json.JSONDecodeError as e:
                    st.error("The output from DevaAi is not valid JSON. Please verify the response format.")
                    st.write(f"Raw content: {crew_output.raw}")
                    st.exception(e)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()
