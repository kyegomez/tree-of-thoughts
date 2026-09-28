"""Enumerative combinatorics: domino tilings of the 8 x 8 board.

The exact count was found independently by Kasteleyn (1961) and by
Temperley and Fisher (1961), using Pfaffians and a product of cosines. Brute
force is hopeless, so the search has to find and apply an exact method.
Candidate steps are sampled independently to explore different methods, and
two evaluator ratings are averaged per candidate. Expected answer:
12,988,816.
"""

from swarms import TreeOfThoughts

tot = TreeOfThoughts(
    name="Combinatorics-Solver",
    model_name="gpt-5.4",
    search_algorithm="bfs",
    generation_strategy="sample",
    evaluation_strategy="value",
    num_thoughts=3,
    breadth=2,
    max_depth=5,
    n_evaluate_samples=2,
    thought_description=(
        "One step toward an exact count: choose a method (a transfer "
        "matrix, a Pfaffian, a product formula), state it precisely, check "
        "it on a small board, or evaluate part of it."
    ),
    evaluation_criteria=(
        "Is the method exact rather than an estimate? Are the formula's "
        "indices and ranges right for an 8 x 8 board? Does the method give "
        "the known small cases: 2 tilings of the 2 x 2 board and 36 of the "
        "4 x 4 board? Is the arithmetic exact?"
    ),
)

answer = tot.run(
    "In how many ways can an 8 x 8 chessboard be tiled by 32 dominoes, "
    "each covering two adjacent squares? Give the exact number and "
    "justify it."
)
print(f"Answer: {answer}\n")

result = tot.last_result
for number, step in enumerate(result.steps, 1):
    print(f"{number}. {step}")
print(
    f"\nsolved={result.solved} nodes_expanded={result.nodes_expanded} "
    f"llm_calls={result.llm_calls} tokens={result.usage['total_tokens']}"
)
