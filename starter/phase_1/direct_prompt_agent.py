# Test script for DirectPromptAgent class

from workflow_agents.base_agents import DirectPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
prompt = "What is the Capital of France?"

# TODO: 3 - Instantiate the DirectPromptAgent as direct_agent
direct_agent = DirectPromptAgent

# TODO: 4 - Use direct_agent to send the prompt defined above and store the response
direct_agent_response = direct_agent(prompt)

# Print the response from the agent
print(direct_agent_response)

print("Knowledge source: response generated directly by the OpenAI GPT-3.5 model via DirectPromptAgent using only the provided prompt.")
