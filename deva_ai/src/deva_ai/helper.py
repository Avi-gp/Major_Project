import os 
from openai import OpenAI
from dotenv import load_dotenv
class helperFunction:

    def __init__(self):
        load_dotenv()
        self.client = OpenAI(
            base_url = "https://integrate.api.nvidia.com/v1",
            api_key= os.getenv("API_KEY_2")
            )

    def call_llm(self, system_prompt, analysis_prompt):
        """
        Sends a request to llama-3.3 with the system and user prompts.
        """
        try:
            response = self.client.chat.completions.create(
                model="nvidia/llama-3.3-nemotron-super-49b-v1",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": analysis_prompt},
                ],
                temperature=0.6,
            )
            
            raw_response = response.choices[0].message.content
            return raw_response
        
        
        except Exception as e:
            print("Error calling LLM:", e)
            return None
