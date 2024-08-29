from tree_of_thoughts import TotAgent, ToTDFSAgent
from dotenv import load_dotenv

load_dotenv()

# Create an instance of the TotAgent class
tot_agent = TotAgent()

# Create an instance of the ToTDFSAgent class with specified parameters
dfs_agent = ToTDFSAgent(
    agent=tot_agent,  # Use the TotAgent instance as the agent for the DFS algorithm
    threshold=0.8,  # Set the threshold for evaluating the quality of thoughts
    max_loops=1,  # Set the maximum number of loops for the DFS algorithm
    prune_threshold=0.5,  # Branches with evaluation < 0.5 will be pruned
    number_of_agents=4,  # Set the number of agents to be used in the DFS algorithm
)

# Define the initial state for the DFS algorithm
initial_state = """

Your task: is to use 4 numbers and basic arithmetic operations (+-*/) to obtain 24 in 1 equation, return only the math

"""

# Run the DFS algorithm to solve the problem and obtain the final thought
final_thought = dfs_agent.run(initial_state)

# Print the final thought in JSON format for easy reading
print(final_thought)
