"""State-space search: missionaries and cannibals.

Each thought is one round trip of the boat, so the eleven crossings of an
optimal plan fit in six steps. Depth-first search commits to the most
promising round trip and backtracks when a bank becomes unsafe or the plan
starts to undo itself. max_expansions caps the cost of backtracking.
Expected answer: 11 crossings.
"""

from swarms import TreeOfThoughts

tot = TreeOfThoughts(
    name="River-Crossing-Planner",
    model_name="gpt-5.4",
    search_algorithm="dfs",
    generation_strategy="propose",
    evaluation_strategy="value",
    num_thoughts=3,
    max_depth=6,
    max_expansions=20,
    thought_description=(
        "One round trip: who crosses to the far bank and who brings the "
        "boat back (or, at the end, the final crossing), with the number "
        "of missionaries and cannibals on each bank afterwards and the "
        "running count of crossings."
    ),
    evaluation_criteria=(
        "Is every crossing legal: one or two people in the boat, and no "
        "bank where missionaries are present but outnumbered? Are the "
        "bank counts right? Does the plan make progress, or does it "
        "repeat an earlier state? Judge whether it can still finish in "
        "the minimum number of crossings."
    ),
)

answer = tot.run(
    "Three missionaries and three cannibals must cross a river. Their boat "
    "holds at most two people and cannot cross empty. If the cannibals on "
    "either bank ever outnumber the missionaries there (while at least "
    "one missionary is on that bank), the missionaries are eaten. People "
    "in the boat count as being on the bank where it has landed. Give a "
    "plan that gets everyone across with the minimum number of crossings, "
    "and state that minimum."
)
print(f"Answer: {answer}\n")

result = tot.last_result
for number, step in enumerate(result.steps, 1):
    print(f"{number}. {step}")
print(
    f"\nsolved={result.solved} nodes_expanded={result.nodes_expanded} "
    f"llm_calls={result.llm_calls} tokens={result.usage['total_tokens']}"
)
