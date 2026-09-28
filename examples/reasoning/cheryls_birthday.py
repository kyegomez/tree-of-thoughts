"""Epistemic reasoning: Cheryl's birthday.

Each statement in the puzzle eliminates dates according to what one person
knows about what the other knows, which single-pass answers often get
wrong. Candidates are compared by vote rather than rated in isolation, and
three votes per comparison keep more than one line of elimination alive.
Expected answer: July 16.
"""

from swarms import TreeOfThoughts

tot = TreeOfThoughts(
    name="Epistemic-Reasoner",
    model_name="gpt-5.4",
    search_algorithm="bfs",
    generation_strategy="propose",
    evaluation_strategy="vote",
    num_thoughts=3,
    breadth=2,
    max_depth=4,
    n_evaluate_samples=3,
    thought_description=(
        "One elimination: take one statement, say what it reveals about "
        "the speaker's knowledge, and list the dates that remain possible."
    ),
    evaluation_criteria=(
        "Does the elimination follow from what the speaker could know, "
        "given only the month or only the day they were told? Are the "
        "statements applied in order, each to the dates left by the "
        "previous one? Is any date wrongly kept or wrongly removed?"
    ),
)

answer = tot.run(
    "Albert and Bernard want to know Cheryl's birthday. She gives them ten "
    "possible dates: May 15, May 16, May 19, June 17, June 18, July 14, "
    "July 16, August 14, August 15 and August 17. She then tells Albert "
    "only the month and Bernard only the day. Albert says: 'I don't know "
    "when Cheryl's birthday is, but I know that Bernard doesn't know "
    "either.' Bernard says: 'At first I didn't know, but now I do.' Albert "
    "says: 'Then I also know.' When is Cheryl's birthday?"
)
print(f"Answer: {answer}\n")

result = tot.last_result
for number, step in enumerate(result.steps, 1):
    print(f"{number}. {step}")
print(
    f"\nsolved={result.solved} nodes_expanded={result.nodes_expanded} "
    f"llm_calls={result.llm_calls} tokens={result.usage['total_tokens']}"
)
