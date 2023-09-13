# Multi-Agent Conversation System

**Author:** Dolvido
**Email:** dolvido@protonmail.com

A multi-agent conversation system that features self-evaluating agents capable of engaging in a multi-turn conversation with the user. Each agent can assess the relevance of its response using the conversation history, thereby maintaining a more focused and coherent conversation.

## Features

- **Multi-Agent Conversation:** Multiple agents participate in a round-robin conversation, responding to the latest input in the conversation.
- **Self-Evaluating Agents:** Agents evaluate the relevance of their responses and can correct themselves if they go off-topic.
- **Conversation Summarization:** Periodic summarization of the conversation to highlight the most relevant responses.

## Installation

To install the necessary dependencies for this project, follow the instructions below:

```bash
pip install openai
```
## Usage
To start a multi-agent conversation, run the multi-turn-conversation.py script:
python multi-turn-conversation.py

You will be prompted to enter the initial query to kickstart the conversation.

## Development
#Project Structure
utilities.py: Contains utility functions for calculating response relevance and for self-evaluation of agent responses.
multi-turn-conversation.py: The main script that initializes the agents and controls the multi-turn conversation.
Adding New Features
To add new features to the agents or the conversation system, modify the RootChakraAgent class and the main script as necessary.

#Testing
To test the system, run the multi-turn-conversation.py script with various queries and observe the agents' responses and the conversation summarizations.

#Contributing
If you would like to contribute to this project, please fork the repository and create a pull request with your changes.

#License
This project is licensed under the MIT License.
