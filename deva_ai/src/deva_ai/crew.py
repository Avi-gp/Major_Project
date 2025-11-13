from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from streamlit import context
from tools.file_handling_tool import FileHandlingTool
from tools.data_preprocessing_tool import DataPreprocessingTool
from tools.feature_engineering_tool import FeatureEngineeringTool
from dotenv import load_dotenv
import os
# Load environment variables
load_dotenv()

# Initialize LLM with API key and model name
llm_model_name = os.getenv('MODEL')
api_key = os.getenv('NVIDIA_NIM_API_KEY_1')

LLM_model = LLM(
    model=llm_model_name,
    api_key=api_key,
    temperature=0.6,
)

# Initialize custom tools
file_handling_tool = FileHandlingTool()
data_preprocessing_tool = DataPreprocessingTool()
feature_engineering_tool = FeatureEngineeringTool()

@CrewBase
class DevaAi:
    """DevaAi crew"""

    # Define paths to YAML configuration files
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def data_ingestion_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['DataIngestionAgent'],  
            tools=[file_handling_tool],
            verbose=True,
            llm=LLM_model
        )
    
    @agent
    def data_preprocessing_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['DataPreprocessingAgent'],
            tools=[data_preprocessing_tool],
            verbose=True,
            llm=LLM_model
        )
    
    @agent
    def feature_engineering_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['FeatureEngineeringAgent'],
            tools=[feature_engineering_tool],
            verbose=True,
            llm=LLM_model
        )

    @task
    def data_ingestion_task(self) -> Task:
        return Task(
            config=self.tasks_config['IngestionTask'],  
            agent=self.data_ingestion_agent(),
            tools=[file_handling_tool],  
            verbose=True
        )
    
    @task
    def data_preprocessing_task(self) -> Task:
        return Task(
            config=self.tasks_config['PreprocessingTask'],
            agent=self.data_preprocessing_agent(),
            tools=[data_preprocessing_tool],
            verbose=True
        )
    
    @task
    def feature_engineering_task(self) -> Task:
        return Task(
            config=self.tasks_config['FeatureEngineeringTask'],
            agent=self.feature_engineering_agent(),
            tools=[feature_engineering_tool],
            context=[self.data_preprocessing_task()],
            input={"preprocessed_file_path": "preprocessed_file_path",
                "preprocessed_file_name": "preprocessed_file_name"},     
            verbose=True
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the DevaAi crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )