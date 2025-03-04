# file_handling_tool.py
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import pandas as pd
import os
import logging
import json

class FilePathInput(BaseModel):
    """Input schema for FileHandlingTool that uses file path."""
    file_path: str = Field(..., description="Absolute path to the uploaded file.")
    file_name: str = Field(..., description="The name of the uploaded file.")

class FileHandlingTool(BaseTool):
    name: str = "File Handling Tool"
    description: str = "Process and analyze dataset files based on provided file path"
    args_schema: Type[BaseModel] = FilePathInput

    def _run(self, file_path: str, file_name: str) -> str:
        """
        Process the file from a file path and provide analysis.
        
        Args:
            file_path (str): Absolute path to the file
            file_name (str): Name of the uploaded file
            
        Returns:
            str: JSON string containing analysis results
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
            
            # Generate analysis
            preview_rows = min(5, len(df))
            
            result = {
                "message": "File analysis completed successfully",
                "dataset_shape": [int(df.shape[0]), int(df.shape[1])],
                "preview": df.head(preview_rows).to_dict('records'),
                "columns": df.columns.tolist(),
                "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()}
            }
            
            # Return as a clean JSON string
            return json.dumps(result)
            
        except Exception as e:
            logging.error(f"Error processing file: {str(e)}")
            error_result = {
                "error": str(e),
                "message": f"Failed to process file at {file_path}"
            }
            return json.dumps(error_result)