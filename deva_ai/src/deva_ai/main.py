import json
import pandas as pd
import streamlit as st
import os
import re
from crew import DevaAi
from utils import save_uploadedfile , extract_data_with_regex , extract_preprocessing_info , preprocess_for_json , parse_json_safely , extract_value
from streamlit_components.data_ingestion_output import display_data_ingestion_results
from streamlit_components.data_preprocessing_output import display_preprocessing_results
#from streamlit_components.feature_engineering_output import display_feature_engineering_results

# Set the page configuration as the first Streamlit command
st.set_page_config(page_title="DEVA AI : AI-Powered Data Analytics", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "Home"

#-----------------------------------------------------------------------------------------------#
def main():
    # Custom CSS for styling - optimized for dark theme
    st.markdown("""
        <style>
            .title {
                font-family: 'Arial', sans-serif;
                text-align: center;
                font-size: 38px;
                color: #62b3ff;
                padding: 20px 0px 10px 0px;
                font-weight: bold;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
            }
            .subheader {
                font-family: 'Arial', sans-serif;
                font-size: 26px;
                color: #a1c9ff;
                padding: 12px 0px;
                border-bottom: 1px solid #444;
                margin-bottom: 18px;
            }
            .card {
                background-color: #1e1e1e;
                border-radius: 10px;
                padding: 22px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.3);
                margin-bottom: 24px;
                border: 1px solid #333;
            }
            .stButton>button {
                background:Transparent
                background-color: #4d8bf0;
                color: white;
                font-family: 'Arial', sans-serif;
                border-radius: 5px;
                padding: 12px 24px;
                border: none;
                font-weight: bold;
                transition: all 0.3s;
            }
            .stButton>button:hover {
                background-color: transparent;
                box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            }
            .footer {
                text-align: center;
                color: #999;
                padding: 25px;
                font-size: 14px;
                border-top: 1px solid #333;
                margin-top: 30px;
            }
            .metric-card {
                background: linear-gradient(145deg, #2d2d2d, #252525);
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                text-align: center;
                margin: 10px 5px;
                border: 1px solid #3a3a3a;
                transition: all 0.3s ease;
            }

            .metric-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 6px 16px rgba(0,0,0,0.4);
                border-color: #4d8bf0;
            }

            .metric-value {
                font-size: 28px;
                font-weight: bold;
                color: #5e9eff;
                text-shadow: 0 2px 4px rgba(0,0,0,0.3);
                margin-bottom: 8px;
            }

            .metric-label {
                font-size: 14px;
                color: #bbb;
                margin-top: 5px;
            }

            /* Improved section headers */
            .subheader {
                font-family: 'Arial', sans-serif;
                font-size: 28px;
                color: #a1c9ff;
                padding: 15px 0px;
                border-bottom: 2px solid #3a3a3a;
                margin-bottom: 25px;
                text-shadow: 0 2px 3px rgba(0,0,0,0.2);
            }

            .highlight-text {
                color: #62b3ff;
                font-weight: 600;
                background: linear-gradient(90deg, rgba(98, 179, 255, 0.1) 0%, rgba(0, 0, 0, 0) 100%);
                padding: 8px 12px;
                border-radius: 5px;
                border-left: 3px solid #62b3ff;
                margin: 15px 0;
            }

            /* Enhanced tables */
            .dataframe {
                border-radius: 8px !important;
                overflow: hidden !important;
                border: 1px solid #3a3a3a !important;
            }

            /* Download button enhancement */
            .stDownloadButton > button {
                background-color: #4d8bf0 !important;
                color: white !important;
                padding: 12px 24px !important;
                font-weight: bold !important;
                border-radius: 8px !important;
                border: none !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                margin: 20px auto !important;
                transition: all 0.3s !important;
            }

            .stDownloadButton > button:hover {
                background-color: #3a78dd !important;
                box-shadow: 0 4px 12px rgba(0,0,0,0.4) !important;
                transform: translateY(-2px) !important;
            }

            /* Error container styling */
            .stException {
                background-color: rgba(187, 51, 51, 0.1) !important;
                border-radius: 8px !important;
                padding: 15px !important;
                border: 1px solid rgba(187, 51, 51, 0.3) !important;
            }

            /* Card hover effect */
            .card {
                transition: all 0.3s ease;
            }

            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 8px 16px rgba(0,0,0,0.4);
            }

            /* Enhanced expander styling */
            .streamlit-expanderHeader {
                background-color: #2d2d2d !important;
                border-radius: 8px !important;
                color: #a1c9ff !important;
                padding: 12px 15px !important;
                font-weight: 600 !important;
                border: 1px solid #3a3a3a !important;
                transition: all 0.2s ease !important;
            }

            .streamlit-expanderHeader:hover {
                background-color: #353535 !important;
                color: #62b3ff !important;
            }

            /* Progress bar enhancement */
            .stProgress > div > div {
                background-color: #4d8bf0 !important;
            }

            /* Spinner enhancement */
            .stSpinner > div {
                border-top-color: #4d8bf0 !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x150.png?text=DEVA+AI", width=150)
        st.markdown("### Navigation")
        if st.button("🏠 Home"):
            st.session_state.page = "Home"
        if st.button("ℹ️ About"):
            st.session_state.page = "About"
        if st.button("❓ Help"):
            st.session_state.page = "Help"
        #page = st.radio("Navigations", ["Home", "About", "Help"])
        
        st.markdown("---")
        st.markdown("### Contact")
        st.markdown("📧 support@devaai.com")
        st.markdown("🌐 www.devaai.com")

    page = st.session_state.page


    # Main content based on selected page
    if page == "Home":
        # Title with icon
        st.markdown('<h1 class="title">🧑‍💼 DEVA AI : AI-Powered Data Analytics </h1>', unsafe_allow_html=True)
        
        # Introduction
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
                <div class="card">
                    <h3 class="highlight-text">Unlock the Power of Your Data</h3>
                    <p class="description-text">Upload your dataset and let our advanced AI transform raw numbers into clear, actionable insights. No coding required – just upload and discover what your data has been trying to tell you.</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="card">
                    <h4 class="highlight-text">Key Benefits</h4>
                    <ul>
                        <li class="benefit-item"><strong>Instant Exploration</strong> – Understand your data in seconds</li>
                        <li class="benefit-item"><strong>Smart Visualizations</strong> – See patterns you might have missed</li>
                        <li class="benefit-item"><strong>Actionable Insights</strong> – Make data-driven decisions with confidence</li>   
                    </ul>
                </div>
            """, unsafe_allow_html=True)

        # Upload file section            
        st.markdown('<h2 class="subheader">Upload Your Dataset</h2>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

        # Update the file upload section in main.py

        if uploaded_file:
            try:
                # Save file and get absolute path
                file_path = save_uploadedfile(uploaded_file)
                
                if file_path:
                    st.success(f"File successfully saved at: {file_path}")
                    
                    # Check file extension
                    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                    if file_extension not in ['.csv', '.xlsx']:
                        st.error("Unsupported file format. Please upload CSV or Excel file.")
                        st.stop()
                    
                    # Create DevaAi instance
                    deva = DevaAi()
                    
                    # Prepare input for CrewAI with file path
                    
                    inputs = {
                        "file_path": file_path,
                        "file_name": uploaded_file.name,
                        "preprocessed_file_path":"",
                        "preprocessed_file_name":""
                    }

                    with st.spinner('Processing dataset... Please wait'):
                        crew_output = deva.crew().kickoff(inputs=inputs)
                        
                    # Process task outputs and display results

                    if hasattr(crew_output, 'tasks_output'):
                        # Display progress
                        progress_bar = st.progress(0)
                        
                        # Task 0: Data Ingestion Results
                        if len(crew_output.tasks_output) > 0:
                            st.success("✅ Data successfully loaded and analyzed!")
                            display_data_ingestion_results(crew_output)
                            progress_bar.progress(50)
                        
                        # Task 1: Preprocessing Results
                        if len(crew_output.tasks_output) > 1:
                            display_preprocessing_results(crew_output.tasks_output[1])
                            progress_bar.progress(75)
                            st.success("✅ Data preprocessing completed!")

                        # Task 2: Feature Engineering Results
                        if len(crew_output.tasks_output) > 2:
                            st.write(crew_output.tasks_output[2].raw)
                            #display_feature_engineering_results(crew_output.tasks_output[2])
                            progress_bar.progress(100)
                            st.success("✅ Feature Engineering completed!")
                        
                        # Clear progress bar after completion
                        progress_bar.empty()
                    else:
                        # Fallback to raw output if tasks_output is not available
                        st.warning("Task output not structured as expected. Displaying raw results:")
                        try:
                            raw_content = crew_output.raw.strip("`")
                            raw_content = preprocess_for_json(raw_content)
                            result = json.loads(raw_content)
                            
                            # Display results using existing code...
                            st.json(result)
                        except Exception as parse_error:
                            st.error(f"Error parsing raw output: {str(parse_error)}")
                            st.write(crew_output.raw)
                else:
                    st.error("Failed to save the uploaded file.")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)
        

    elif page == "About":
        st.markdown('<h1 class="title">About DEVA AI</h1>', unsafe_allow_html=True)
        st.markdown("""
            <div class="card">
                <h3 class="highlight-text">Our Mission</h3>
                <p class="description-text">DEVA AI is on a mission to democratize data analysis and make it accessible to everyone, 
                regardless of their technical background. We believe that the insights hidden in data should be available to all, 
                not just those with specialized training.</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="card">
                <h3 class="highlight-text">Our Technology</h3>
                <p class="description-text">Powered by cutting-edge AI models and the innovative CrewAI framework, DEVA AI transforms 
                complex data analysis into an intuitive experience. Our platform bridges the gap between raw data and meaningful insights, 
                eliminating the steep learning curve traditionally associated with data science.</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="card">
                <h3 class="highlight-text">Our Team</h3>
                <p class="description-text">Behind DEVA AI is a diverse team of data scientists, AI researchers, and software engineers 
                united by a shared vision: making data analysis intuitive, powerful, and accessible. With decades of combined experience 
                in AI and data science, we're committed to continuously improving how people interact with and understand their data.</p>
            </div>
        """, unsafe_allow_html=True)
        

        
    elif page == "Help":
        st.markdown('<h1 class="title">Help & Documentation</h1>', unsafe_allow_html=True)
        
        with st.expander("How to Use DEVA AI"):
            st.markdown("""
                <p class="description-text">
                <ol>
                    <li><strong class="highlight-text">Upload your data</strong>: Click on the file upload button on the home page and select a CSV or Excel file from your computer.</li>
                    <li><strong class="highlight-text">Processing</strong>: Our AI engine will automatically analyze your data, identifying patterns and preparing visualizations.</li>
                    <li><strong class="highlight-text">Explore insights</strong>: Navigate through the generated sections to discover different perspectives on your data.</li>
                    <li><strong class="highlight-text">Download results</strong>: Save any visualizations or insights for your presentations, reports, or further analysis.</li>
                </ol>
                </p>
            """, unsafe_allow_html=True)
        
        with st.expander("Supported File Formats"):
            st.markdown("""
                <p class="description-text">
                DEVA AI currently supports:
                <ul>
                    <li><strong class="highlight-text">CSV Files</strong> (.csv) - Comma-separated values</li>
                    <li><strong class="highlight-text">Excel Files</strong> (.xlsx) - Microsoft Excel spreadsheets</li>
                </ul>
                
                Our development roadmap includes support for JSON, SQL databases, and direct API connections in upcoming releases.
                </p>
            """, unsafe_allow_html=True)
        
        with st.expander("Troubleshooting Common Issues"):
            st.markdown("""
                <p class="description-text">
                <strong class="highlight-text">File Upload Problems</strong>
                <ul>
                    <li>Ensure your file is in CSV or Excel format with a valid extension</li>
                    <li>Check that the file size is under 200MB</li>
                    <li>Verify that your file is not corrupted by opening it in another application first</li>
                </ul>
                
                <strong class="highlight-text">Processing Delays</strong>
                <ul>
                    <li>Large datasets (>100,000 rows) may require additional processing time</li>
                    <li>Files with many columns (>50) might take longer to analyze</li>
                    <li>Complex data types or mixed formats can extend processing duration</li>
                </ul>
                
                <strong class="highlight-text">Visualization Issues</strong>
                <ul>
                    <li>Ensure your dataset contains sufficient numeric columns for statistical analysis</li>
                    <li>Check for missing values that might affect visualization quality</li>
                    <li>Consider using a more balanced dataset if your data is heavily skewed</li>
                </ul>
                </p>
            """, unsafe_allow_html=True)
        
        with st.expander("Contact Support"):
            st.markdown("""
                <p class="description-text">
                Our support team is ready to help with any questions or issues you encounter:
                
                <div style="margin-top: 15px; margin-bottom: 15px;">
                    <strong class="highlight-text">📧 Email Support:</strong> support@devaai.com<br>
                    <strong class="highlight-text">🌐 Support Portal:</strong> www.devaai.com/support<br>
                    <strong class="highlight-text">💬 Live Chat:</strong> Available on our website during business hours (9AM-6PM EST)
                </div>
                
                We aim to respond to all inquiries within 24 hours, with priority support available for enterprise customers.
                </p>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
        <div class="footer">
            © 2025 DEVA AI - Transforming Data into Insights | <a href="#">Privacy Policy</a> | <a href="#">Terms of Service</a>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
