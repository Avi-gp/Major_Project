# file_upload_tool.py
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import pandas as pd
import os
import logging

class FileUploadToolInput(BaseModel):
    """Input schema for FileUploadTool."""
    file_path: str = Field(..., description="The path to the uploaded file.")
    file_name: str = Field(..., description="The name of the uploaded file.")

class FileUploadTool(BaseTool):
    name: str = "File Upload Tool"
    description: str = "Process and analyze dataset files"
    args_schema: Type[BaseModel] = FileUploadToolInput

    def _run(self, file_path: str, file_name: str) -> dict:
        """
        Process the uploaded file and return analysis results.
        
        Args:
            file_path (str): Path to the uploaded file
            file_name (str): Name of the uploaded file
            
        Returns:
            dict: Analysis results including shape, preview, and statistics
        """
        try:
            # Read the dataset
            if file_name.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_name.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format. Please upload CSV or Excel file.")
                
            # Get numeric statistics if numeric columns exist
            numeric_df = df.select_dtypes(include=['int64', 'float64'])
            numeric_stats = None
            if not numeric_df.empty:
                numeric_stats = numeric_df.describe().to_dict()
                
            return {
                "message": "File processed successfully",
                "shape": df.shape,
                "preview": df.head().to_dict('records'),
                "numeric_stats": numeric_stats
            }
            
        except Exception as e:
            logging.error(f"Error processing file: {str(e)}")
            raise e