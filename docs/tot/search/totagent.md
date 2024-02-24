
# Tree of Thoughts (ToT) Library Documentation

## Introduction

The Tree of Thoughts (ToT) Library serves as a Python implementation wrapper for the OpenAI Language Model API, aimed at enhancing intelligent agent capabilities. Its primary objective is the generation and evaluation of textual thoughts in the form of prompt-based dialogues. This library introduces the `ToTAgent` class, which encapsulates various methods for thought generation and solution evaluation.

`ToTAgent` integrates closely with agent classes from swarms, another framework for cooperative problem-solving with agents. The purpose of `ToTAgent` is to provide an interface that leverages the cognitive capacities of language models for reasoning and problem-solving tasks.

## Architecture Overview

The ToT library centers around the `ToTAgent` class, which wraps a language model API to perform specific tasks:
- **Generate Thoughts**: Produce different thoughts based on the given state and initial prompts.
- **Generate Solution**: Concoct a singular solution from a state with consideration of previously rejected solutions.
- **Evaluate States**: Appraise states of reasoning to facilitate the decision-making process.

This functionality is encapsulated using a strategy pattern, allowing the `ToTAgent` to be configured with different generation and evaluation strategies.

## ToTAgent Class

The ToTAgent class interfaces with an OpenAI Language Model to deliver the above functionalities in the form of methods. Below is the class signature and its corresponding arguments:

| Argument             | Type     | Description                                           | Default     |
|----------------------|----------|-------------------------------------------------------|-------------|
| `agent`              | `Agent`  | The agent class instance from swarms framework.       | Required    |
| `strategy`           | `str`    | The generation strategy for thoughts.                 | `"cot"`     |
| `evaluation_strategy`| `str`    | The evaluation strategy for states.                   | `"value"`   |
| `enable_react`       | `bool`   | Toggle for enabling ReAct prompting.                  | `True`      |
| `k`                  | `int`    | Number of thoughts to generate per iteration.        | `3`         |

### Methods

| Method                                                      | Returns | Description                                                                       |
|-------------------------------------------------------------|---------|-----------------------------------------------------------------------------------|
| `run(task: str)`                                            | `list`  | Generate a list of thoughts based on the provided task using the OpenAI API.      |
| `generate_thoughts(state, k, initial_prompt, rejected_solutions=None)` | `list`  | Generate a list of thoughts from the state considering rejected solutions.        |
| `generate_solution(initial_prompt, state, rejected_solutions=None)`    | `str`   | Generate a single solution from the state considering rejected solutions.         |
| `evaluate_states(states, initial_prompt)`                   | `dict`  | Evaluate states to determine their effectiveness in the context of given prompts.  |

## Usage Examples

### Example 1: Basic Usage

```python
from tree_of_thoughts.tot_agent import ToTAgent
from swarms import Agent

# Create an instance of the swarms Agent class.
agent = Agent()

# Instantiate a ToTAgent with default settings.
model = ToTAgent(agent)

# Run the ToTAgent to generate thoughts.
task = "Suggest innovative features for a new smartphone model."
thoughts = model.run(task)
print(thoughts)
```

### Example 2: Generating Thoughts with Custom Strategy

```python
from tree_of_thoughts.tot_agent import ToTAgent
from swarms import Agent

# Initialize the swarms Agent
agent = Agent()

# Create a ToTAgent with a custom strategy for thought generation.
model = ToTAgent(agent, strategy="advanced_cot")

# Define the task and generate thoughts using the custom strategy.
task = "Plan a birthday party with a surprise element."
thoughts = model.generate_thoughts(state="Initial Plan", k=5, initial_prompt=task)
print(thoughts)
```

### Example 3: Evaluating States with 'vote' Strategy

```python
from tree_of_thoughts.tot_agent import ToTAgent
from swarms import Agent

# Set up the swarms Agent.
agent = Agent()

# Instantiate a ToTAgent with the evaluation strategy set to 'vote'.
model = ToTAgent(agent, evaluation_strategy="vote")

# Prepare a list of states to evaluate.
states = ["State 1:...", "State 2:...", "State 3:..."]
initial_prompt = "Improve user experience on a website."

# Evaluate states using the 'vote' strategy.
state_values = model.evaluate_states(states, initial_prompt)
print(state_values)
```

## Additional Information and Tips

It is essential to understand that the ToTAgent class uses an Agent class instance that must be defined in the swarms framework. The Agent class should implement the actual communication with the OpenAI API.

When using the `run` method, ensure that tasks are formatted appropriately to comply with the expectations of the underlying language model API.

The `evaluate_states` method's behavior will significantly vary based on the strategy selected (`"value"` or `"vote"`). The choice of strategy should align with the specific evaluation needs of the reasoning tasks.

## References and Resources

The ToT library is an innovation layer atop OpenAI's Language Model API. It is designed to provide an abstraction that simplifies the complexity inherent in advanced natural language processing tasks. For more in-depth knowledge, users can explore the following resources:
- OpenAI API Documentation: [Official OpenAI API Docs](https://beta.openai.com/docs/)
- Swarms Framework: A lookup to the related swarms framework may be essential to know how to set up the Agent instance required by ToTAgent.

This comprehensive guide aims to jumpstart your integration with ToTAgent, enabling powerful reasoning and problem-solving capabilities in your applications.
