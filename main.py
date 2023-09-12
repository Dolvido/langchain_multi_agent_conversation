# my imports 
from utilities import calculate_relevance_score, self_evaluating_agent

from langchain.llms import OpenAI
from langchain import PromptTemplate, ConversationChain
from langchain.chains import LLMChain
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.agents.load_tools import get_all_tool_names
from langchain.chat_models import ChatOpenAI

import pandas as pd  # We will use pandas to create a DataFrame from the conversation history


llm = OpenAI()

get_all_tool_names()
tools = load_tools(["wikipedia", "llm-math"], llm=llm)

class ConversationAgent:
    def __init__(self, name, tools, llm, strategy='react_description', parameters=None):
        self.name = name
        self.strategy = strategy  # New attribute to hold the strategy
        self.parameters = parameters if parameters else {}  # New attribute to hold various parameters
        self.agent = initialize_agent(
            tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
        )
        
    def respond(self, query):
        # We will modify this method to generate responses based on the strategy attribute
        if self.strategy == 'react_description':
            response = self.agent.run(query)
        elif self.strategy == 'template_based':
            response = self.template_based_response(query)
        # ... (other strategies can be added here)
        else:
            response = "Unknown strategy"

        # evaluate and potentially refine response
        refined_response = self_evaluating_agent(response, query)
            
        return refined_response

    def template_based_response(self, query):
        # Placeholder for template-based response generation
        return "Template-based response"


agents = [ConversationAgent(name=f"Agent_{i}", tools=tools, llm=llm) for i in range(3)]

# Multi-turn conversation
conversation_history = []
initial_query = input("Initial query: ")  # Getting the initial query

for i in range(5):  # 5 turns of conversation
    for agent in agents:
        response = agent.respond(initial_query)  # Using the last input as the new input
        print(f"{agent.name} responds: {response}")
        conversation_history.append((agent.name, initial_query, response))
        initial_query = response  # Updating the initial_query with the new response
    
    # Summarizing the conversation every few turns (here, after each round of conversation)
    if (i+1) % 2 == 0:
        # Select the most relevant responses for the summary
        relevant_responses = [
            resp for _, _, resp in conversation_history[-(len(agents)*2):]
            if calculate_relevance_score(initial_query, resp) > 0.9
        ]
        summary = " ".join(relevant_responses)
        print(f"Summarizing the last 2 rounds: {summary}")

    
    # Calculate the relevance score of the current turn
        relevance_score = calculate_relevance_score(initial_query, response)
        print(f"Relevance score: {relevance_score}")

    

# Display the full conversation history at the end
for i, (agent_name, input_query, resp) in enumerate(conversation_history):
    print(f"Turn {i+1} - {agent_name}: {resp} (in response to: {input_query})")
