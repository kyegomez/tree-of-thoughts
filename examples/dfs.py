from tree_of_thoughts import TotAgent, ToTDFSAgent

tot_agent = TotAgent()

# Create the ToTDFSAgent class with a threshold, max steps, and pruning threshold
dfs_agent = ToTDFSAgent(
    agent=tot_agent,
    threshold=0.8,
    max_loops=1,
    prune_threshold=0.5,  # Branches with evaluation < 0.5 will be pruned
    number_of_agents=4,
)

# Starting state for the DFS algorithm
initial_state = """

Your task: is to use 4 numbers and basic arithmetic operations (+-*/) to obtain 24 in 1 equation, return only the math

"""

# Run the DFS algorithm to solve the problem
final_thought = dfs_agent.run(initial_state)

# Outputs json which is easy to read
print(final_thought)
