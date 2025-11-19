
import os
from dotenv import load_dotenv
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent, RoutingAgent

# Load environment variables from .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

persona = "You are a college professor"

knowledge = "You know everything about Texas"
texas_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, knowledge)

knowledge_europe = "You know everything about Europe"
europe_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, knowledge_europe)

persona_math = "You are a college math professor"
knowledge_math = "You know everything about math, you take prompts with numbers, extract math formulas, and show the answer without explanation"
maths_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona_math, knowledge_math)

routing_agent = RoutingAgent(openai_api_key, {})
agents = [
    {
        "name": "texas agent",
        "description": "Answer a question about Texas",
        "func": lambda x: texas_agent.respond(x)
    },
    {
        "name": "europe agent",
        "description": "Answer a question about Europe",
        "func": lambda x: europe_agent.respond(x)
    },
    {
        "name": "math agent",
        "description": "When a prompt contains numbers, respond with a math formula",
        "func": lambda x: maths_agent.respond(x)
    }
]

routing_agent.agents = agents

prompts = [
    "Tell me about the history of Rome, Texas",
    "Tell me about the history of Rome, Italy",
    "One story takes 2 days, and there are 20 stories",
]

for prompt in prompts:
    print(f"Prompt: {prompt}")
    response = routing_agent.route(prompt)
    print(f"Response:\n{response}\n")
