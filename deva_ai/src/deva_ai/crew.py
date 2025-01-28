from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from tools.file_upload_tool import FileUploadTool
from dotenv import load_dotenv
import os
import yaml

# Load environment variables
load_dotenv()

# Initialize LLM with API key and model name
llm_model_name = os.getenv('MODEL')
api_key = os.getenv('NVIDIA_NIM_API_KEY')

LLM_model = LLM(
    model=llm_model_name,
    api_key=api_key,
    temperature=0.5,
)

# Initialize custom tool
file_upload_tool = FileUploadTool()


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
            tools=[file_upload_tool],
            verbose=True,
            llm=LLM_model
        )

    @task
    def data_ingestion_task(self) -> Task:
        return Task(
            config=self.tasks_config['IngestionTask'],  
            agent=self.data_ingestion_agent(),
			tools=[file_upload_tool],  
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
