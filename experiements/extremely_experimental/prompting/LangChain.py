from abc import abstractmethod, ABC
from langchain import OpenAI, Wikipedia
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.agents.react.base import DocstoreExplorer
from langchain.prompts import StringPromptTemplate
from pydantic import BaseModel, validator


class AbstractLanguageModel(ABC):
    @abstractmethod
    def generate_thoughts(self, state, k):
        pass

    @abstractmethod
    def evaluate_states(self, states):
        pass


class CustomLanguageModel(AbstractLanguageModel):
    def __init__(self, model):
        self.model = model

    def generate_thoughts(self, state, k):
        #implement the thought generation logic using self.model
        pass

    def evaluate_states(self, states):
        #implement state evaluation logic using self.model
        pass

class LangchainCustomLanguageModel(AbstractLanguageModel):
    def __init__(self, model):
        self.model = model
        docstore = DocstoreExplorer(Wikipedia())
        tools = [
            Tool(
                name="Search",
                func=docstore.search,
                description="useful for when you need to ask with search"
            ),
            Tool(
                name="Lookup",
                func=docstore.lookup,
                description="useful for when you need to ask with lookup"
            )
        ]
        self.agent = initialize_agent(tools, model, agent=AgentType.REACT_DOCSTORE, verbose=True)

    def generate_thoughts(self, state, k):
        state_text = ' '.join(state)
        prompt = f"Given the current state of reasoning: '{state_text}', generate {k} coherent thoughts to continue the reasoning process:"
        response = self.agent.arun(input=prompt)
        thoughts = response.strip().split('\n')
        return thoughts

    def evaluate_states(self, states):
        state_values = {}
        for state in states:
            state_text = ' '.join(state)
            prompt = f"Given the following states of reasoning, vote for the best state:\n{state_text}\n\nVote, and NOTHING ELSE:"
            response = self.agent.arun(input=prompt)
            try:
                value = float(response)
                print(f"value: {value}")
            except ValueError:
                value = 0  # Assign a default value if the conversion fails
            state_values[state] = value
        return state_values