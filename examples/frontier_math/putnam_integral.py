"""Analysis: a closed-form definite integral from the Putnam competition.

Putnam 2005 A5. The integrand has no elementary antiderivative, so the
problem is solved by a well-chosen substitution and a symmetry of the
integrand. Breadth-first search keeps two lines of attack open, so a
substitution that leads nowhere does not end the search. Expected answer:
(pi / 8) ln 2, about 0.2722.
"""

from swarms import TreeOfThoughts

tot = TreeOfThoughts(
    name="Analysis-Solver",
    model_name="gpt-5.4",
    search_algorithm="bfs",
    generation_strategy="propose",
    evaluation_strategy="value",
    num_thoughts=3,
    breadth=2,
    max_depth=5,
    thought_description=(
        "One step of the evaluation: a substitution with its new limits "
        "and differential, a trigonometric or logarithmic identity, a "
        "symmetry of the integral, or solving for the integral."
    ),
    evaluation_criteria=(
        "Is each substitution correct, including its limits and "
        "differential? Is every identity valid on the whole interval? Is "
        "any symmetry argument sound? Check a proposed closed form "
        "against a numerical estimate of the integral."
    ),
)

answer = tot.run(
    "Evaluate the definite integral of ln(1 + x) / (1 + x^2) from x = 0 "
    "to x = 1 in closed form."
)
print(f"Answer: {answer}\n")

result = tot.last_result
for number, step in enumerate(result.steps, 1):
    print(f"{number}. {step}")
print(
    f"\nsolved={result.solved} nodes_expanded={result.nodes_expanded} "
    f"llm_calls={result.llm_calls} tokens={result.usage['total_tokens']}"
)
