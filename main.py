

last_response = None

def generate_next_response(agents, initial_query):
    responses = []
    relevance_scores = []
    for agent in agents:
        response = agent.respond(initial_query)
        responses.append(response)
        relevance_scores.append(calculate_relevance_score(initial_query, response))

    # Calculate the weights for each response.
    weights = relevance_scores / sum(relevance_scores)

    # Generate the next response by averaging the weighted responses of the agents.
    next_response = ' '.join([responses[i] for i in np.argmax(weights)])

    # If the next response is the same as the previous response, return a default response.
    if next_response == last_response:
        next_response = "I'm not sure what else to say. Can you please ask me something else?"
    last_response = next_response

    return next_response


 
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
math = load_tools(["llm-math"], llm=OpenAI(model_name="text-davinci-003"))

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
        self.backup_agent = initialize_agent(
            tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
        )

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
        else:
            response = "Unknown strategy"

    def template_based_response(self, query):
        # Here we are using one of the prompt templates to format the query
        formatted_query = self.prompt_template1.format(product=query)
        response = self.llm_davinci(formatted_query)
        return response

    def chain_based_response(self, query):
        # Here we are using one of the chains to process the query
        response = self.chain.run(query)
        return response

    def gpt_based_response(self, query):
        response = self.conversation.predict(input=query)
        return response
    
class ConversationEvaluatingAgent(ConversationAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize the self-evaluating agent.
        self.self_evaluating_agent = self_evaluating_agent.SelfEvaluatingAgent()

    def respond(self, query):
        # Generate a response using the self-evaluating agent.
        response = self.self_evaluating_agent.generate_response(query)

        # Evaluate the response using the self-evaluating agent.
        evaluation = self.self_evaluating_agent.evaluate_response(response)

        # If the response is not evaluated to be good, generate a new response using one of the other agents in the list.
        if evaluation is not None and evaluation < 0.5:
            response = super().respond(query)

        return response

#agents = [ConversationAgent(name=f"Agent_{i}", tools=tools, llm=llm) for i in range(3)]

agent1 = ConversationAgent(name="Agent_1", tools=tools, llm=llm, strategy='react_description')
agent2 = ConversationAgent(name="Agent_2", tools=tools, llm=llm, strategy='react_description')
agent3 = ConversationAgent(name="GPT_Agent", tools=tools, llm=llm, strategy='react_description')

agents = [agent1, agent2, agent3]

# Multi-turn conversation
conversation_history = []
#initial_query = input("Initial query: ")  # Getting the initial query
initial_query = ""

for i in range(5):  # 5 turns of conversation
    next_response = generate_next_response(agents, initial_query)
        
    print(f"Next response: {next_response}")
    conversation_history.append(("System", initial_query, next_response))
    initial_query = next_response  # Updating the initial_query with the new response

    try:
        # Summarizing the conversation every few turns (here, after each round of conversation)
        if (i+1) % 2 == 0:
            # Select the most relevant responses for the summary
            relevant_responses = [
                resp for _, _, resp in conversation_history[-2:]
                if calculate_relevance_score(initial_query, resp) > 0.9
            ]
            summary = " ".join(relevant_responses)
            print(f"Summarizing the last 2 rounds: {summary}")
    except Exception as e:
        print(f"Error while summarizing: {e}")
        continue


# Display the full conversation history at the end
for i, (agent_name, input_query, resp) in enumerate(conversation_history):
    print(f"Turn {i+1} - {agent_name}: {resp} (in response to: {input_query})")

# Save the conversation history to a CSV file
df = pd.DataFrame(conversation_history, columns=["Agent", "Input", "Response"])
df.to_csv("conversation_history.csv", index=False)
