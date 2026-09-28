"""Diophantine equations: the integer points on y^2 = x^3 - 2.

Fermat claimed and Euler proved that (3, 5) and (3, -5) are the only integer
solutions. The proof works in the ring Z[sqrt(-2)], and every step needs a
justification a number theorist would accept. Depth-first search with a
strict threshold prunes any doubtful step and backtracks to another line of
argument; a custom system prompt gives every call a number theorist's
persona. Expected answer: (x, y) = (3, 5) and (3, -5), with a proof that
there are no others.
"""

from swarms import TreeOfThoughts

tot = TreeOfThoughts(
    name="Number-Theorist",
    model_name="gpt-5.4",
    system_prompt=(
        "You are a research number theorist. Every claim you make follows "
        "from earlier steps or from a standard, named theorem, and you "
        "never treat a numerical search as a proof."
    ),
    search_algorithm="dfs",
    generation_strategy="propose",
    evaluation_strategy="value",
    num_thoughts=2,
    max_depth=6,
    value_threshold=0.7,
    thought_description=(
        "One step of the proof: a single precise deduction together with "
        "its justification. The last step lists every solution and "
        "concludes the proof."
    ),
    evaluation_criteria=(
        "Does the step follow rigorously from the earlier steps or a "
        "standard theorem? If the proof uses an algebraic number ring, is "
        "every property it relies on (unique factorization, the units, "
        "coprimality of the factors) stated and justified? Reject "
        "unjustified claims and numerical evidence offered as proof."
    ),
)

answer = tot.run(
    "Find all integer solutions (x, y) of the equation y^2 = x^3 - 2, and "
    "prove that there are no others."
)
print(f"Answer:\n{answer}\n")

result = tot.last_result
print(
    f"solved={result.solved} nodes_expanded={result.nodes_expanded} "
    f"llm_calls={result.llm_calls} tokens={result.usage['total_tokens']}"
)
