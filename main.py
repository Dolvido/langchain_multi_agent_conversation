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
        self.strategy = strategy
        self.parameters = parameters if parameters else {}

        # Initialize various components
        self.llm_davinci = OpenAI(model_name="text-davinci-003")  # Different LLM
        self.llm_standard = OpenAI()  # Standard LLM for other strategies

        # Initialize prompt templates
        self.prompt_template = PromptTemplate(
            input_variables=["topic"], 
            template="What is a good thing to talk about that reminds you of {topic}?"
        )

        # Initialize chains
        self.chain = LLMChain(llm=self.llm_standard, prompt=self.prompt_template)

        # Initialize agents with different agent types
        self.agent_type1 = initialize_agent(
            tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
        )
        # (Consider initializing other agent types as needed)

        # Initialize conversation chain with memory
        self.conversation = ConversationChain(llm=llm, verbose=True)

        # Initialize chat model
        self.chat_model = ChatOpenAI()

        
    def respond(self, query):
        if self.strategy == 'react_description':
            response = self.agent_type1.run(query)
        elif self.strategy == 'template_based':
            response = self.template_based_response(query)
        elif self.strategy == 'chain_based':
            response = self.chain_based_response(query)
        elif self.strategy == 'gpt_based':
            response = self.gpt_based_response(query)
        # ... (other strategies can be added here)
        else:
            response = "Unknown strategy"

    def template_based_response(self, query):
        # Here we are using one of the prompt templates to format the query
        formatted_query = self.prompt_template1.format(product=query)
        response = self.llm_davinci(formatted_query)
        return response

    def chain_based_response(self, query):
        # Here we are using one of the chains to process the query
        response = self.chain1.run(query)
        return response

    def gpt_based_response(self, query):
        # ... (existing code for GPT-based strategy)
        return response


#agents = [ConversationAgent(name=f"Agent_{i}", tools=tools, llm=llm) for i in range(3)]

agent1 = ConversationAgent(name="Agent_1", tools=tools, llm=llm, strategy='react_description')
agent2 = ConversationAgent(name="Agent_2", tools=tools, llm=llm, strategy='gpt_based')

agents = [agent1, agent2]

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
