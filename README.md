![Tree of Thoughts Banner](images/treeofthoughts.png)

![Discord](https://img.shields.io/discord/999382051935506503)
[![Twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Ftree-of-thoughts)](https://twitter.com/intent/tweet?text=Check%20out%20this%20amazing%20project%20on%20improving%20AI%20reasoning%20-%20Tree%20of%20Thoughts!%20https://github.com/kyegomez/tree-of-thoughts)
[![LinkedIn](https://img.shields.io/badge/Share-LinkedIn-blue?style=social&logo=linkedin)](https://www.linkedin.com/sharing/share-offsite/?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Ftree-of-thoughts)
[![Facebook](https://img.shields.io/badge/Share-Facebook-blue?style=social&logo=facebook)](https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2Ftree-of-thoughts)
[![Reddit](https://img.shields.io/badge/Share-Reddit-orange?style=social&logo=reddit)](https://www.reddit.com/submit?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Ftree-of-thoughts&title=Check%20out%20this%20amazing%20project%20on%20improving%20AI%20reasoning%20-%20Tree%20of%20Thoughts%21)
[![Hacker News](https://img.shields.io/badge/Share-Hacker%20News-orange?style=social&logo=y-combinator)](https://news.ycombinator.com/submitlink?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2Ftree-of-thoughts&t=Check%20out%20this%20amazing%20project%20on%20improving%20AI%20reasoning%20-%20Tree%20of%20Thoughts%21)
[![Pinterest](https://img.shields.io/badge/Share-Pinterest-red?style=social&logo=pinterest)](https://pinterest.com/pin/create/button/?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Ftree-of-thoughts&media=https%3A%2F%2Fgithub.com%2Fkyegomez%2Ftree-of-thoughts%2Fraw%2Fmain%2Ftree-of-thoughts.jpeg&description=Check%20out%20this%20amazing%20project%20on%20improving%20AI%20reasoning%20-%20Tree%20of%20Thoughts%21)
[![WhatsApp](https://img.shields.io/badge/Share-WhatsApp-green?style=social&logo=whatsapp)](https://api.whatsapp.com/send?text=Check%20out%20this%20amazing%20project%20on%20improving%20AI%20reasoning%20-%20Tree%20of%20Thoughts%21%20https%3A%2F%2Fgithub.com%2Fkyegomez%2Ftree-of-thoughts)

# Tree of Thoughts

**[Paper](https://arxiv.org/abs/2305.10601)** · **[Authors' implementation](https://github.com/princeton-nlp/tree-of-thought-llm)** · **[Swarms](https://github.com/kyegomez/swarms)**

Tree of Thoughts (ToT) makes a language model reason by search instead of in a single pass. The model proposes several candidate next steps, scores each one, prunes the weak branches, and backtracks when a line of reasoning fails. In the paper, GPT-4 with chain-of-thought prompting solved 4% of Game of 24 puzzles; with Tree of Thoughts it solved 74%.

> [!NOTE]
> Tree of Thoughts now ships as part of the [Swarms](https://github.com/kyegomez/swarms) framework as `TreeOfThoughts`. The examples and docs below use that implementation.


## How It Works

`TreeOfThoughts` grows a tree of partial solutions:

1. **Generate:** from a node, propose `num_thoughts` candidate next steps, either all in one call (`"propose"`) or one call per step (`"sample"`).
2. **Evaluate:** score each candidate from 0 to 1, either on its own (`"value"`) or by comparing candidates and voting (`"vote"`).
3. **Search:** explore breadth-first with a beam (`"bfs"`) or depth-first with backtracking (`"dfs"`), pruning candidates that score below `value_threshold`.
4. **Answer:** write the final answer from the best path found.

Every model output is a function call validated against a Pydantic schema, so the search never parses free-form prose. Calls at the same level of the tree run concurrently.

## Install

```bash
pip3 install -U swarms
```

Add your API key to a `.env` file in your working directory. Swarms loads it on import.

```bash
OPENAI_API_KEY="your_openai_api_key"
WORKSPACE_DIR="agent_workspace"
```

`model_name` accepts any [LiteLLM](https://docs.litellm.ai/docs/providers) model that supports function calling, so you can set `ANTHROPIC_API_KEY`, `GROQ_API_KEY` or another provider's key instead.

## Quickstart

```python
from swarms import TreeOfThoughts

agent = TreeOfThoughts(
    model_name="gpt-5.4",
    search_algorithm="dfs",
    max_depth=3,
    thought_description="One arithmetic operation on two of the remaining numbers.",
    evaluation_criteria="Can the remaining numbers still reach 24?",
)

answer = agent.run("Use 4, 9, 10 and 13 with + - * / to make 24.")
print(answer)
```

`run(task)` returns the answer. The full search is kept on `agent.last_result`:

```python
result = agent.last_result

for number, step in enumerate(result.steps, 1):  # the best reasoning path
    print(f"{number}. {step}")

print(result.solved)                  # True if a final step cleared value_threshold
print(result.nodes_expanded)          # nodes that had candidates generated
print(result.llm_calls)               # model calls, including the final answer
print(result.usage["total_tokens"])   # token usage for this search
tree = result.to_dict()               # the whole tree as JSON-serializable data
```

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `"gpt-5.4"` | Any LiteLLM model string. The model must support function calling. |
| `search_algorithm` | `"bfs"` | `"bfs"` keeps the best `breadth` nodes per level. `"dfs"` follows the best child first and backtracks. |
| `generation_strategy` | `"propose"` | `"propose"` asks for all candidates in one call. `"sample"` makes one independent call per candidate. |
| `evaluation_strategy` | `"value"` | `"value"` rates each candidate on its own. `"vote"` compares candidates and scores them by votes. |
| `num_thoughts` | `3` | Candidate steps generated per expanded node. |
| `breadth` | `2` | BFS beam width. Ignored by DFS. |
| `max_depth` | `3` | Maximum steps on a path. Steps at this depth must complete the task. |
| `n_evaluate_samples` | `1` | Evaluator calls per candidate (value) or per comparison (vote). More samples give steadier scores. |
| `value_threshold` | `0.5` | Candidates scoring below this are pruned. |
| `max_expansions` | `None` | Cap on nodes expanded per search, to bound cost. |
| `thought_description` | `None` | What one step looks like for your task. Shown to the generator. |
| `evaluation_criteria` | `None` | How to judge progress for your task. Shown to the evaluator. |
| `system_prompt` | built-in | Replace it to give every call a domain persona. |
| `temperature` | `None` | Sampling temperature. `None` uses the provider's default. |
| `max_workers` | `8` | Maximum concurrent model calls. |
| `output_type` | `"final"` | How `run` formats its output. `"final"` returns the answer string. |
| `verbose` | `False` | Log every evaluated candidate. |
| `agent_kwargs` | `None` | Extra `Agent` arguments for every call, such as `max_tokens`, `llm_api_key` or `llm_base_url`. |

### Choosing settings

| Setting | Use | When |
|---|---|---|
| `search_algorithm` | `"bfs"` | Several partial solutions are worth keeping at once (scheduling, multi-step calculations). |
| | `"dfs"` | Case analysis and planning, where you commit to a line and backtrack on a contradiction. |
| `generation_strategy` | `"propose"` | Constrained steps, where one call can list distinct options. |
| | `"sample"` | Open-ended steps, where independent calls give more variety. |
| `evaluation_strategy` | `"value"` | Steps can be checked on their own (arithmetic, logic, units). |
| | `"vote"` | Quality is relative, so comparing candidates beats rating them (estimation, writing). |

`thought_description` and `evaluation_criteria` adapt the search to a domain more than any other setting. Cost grows with `num_thoughts`, `breadth`, `max_depth` and `n_evaluate_samples`; use `max_expansions` to cap it.

## Examples

| Example | Search | Task |
|---|---|---|
| [examples/bfs.py](examples/bfs.py) | BFS, beam of 3 | Game of 24 |
| [examples/dfs.py](examples/dfs.py) | DFS with backtracking, `max_expansions` cap | Game of 24 |

```bash
python examples/dfs.py
```

More examples in mathematics, physics and logic, each with a checkable answer, are in the [Swarms Tree of Thoughts examples](https://github.com/kyegomez/swarms/tree/master/examples/reasoning_agents/tree_of_thoughts_examples).

## Prompts

You can also get Tree of Thoughts-style reasoning from a single prompt, with no code. Paste one of these into any chat model and put your question at the end.

### 1. Step-by-step experts

```txt
Imagine three different experts are answering this question. All experts will
write down 1 step of their thinking, then share it with the group. Then all
experts will go on to the next step, etc. If any expert realises they're wrong
at any point then they leave. The question is...
```

### 2. Collaborative experts

```txt
Simulate three brilliant, logical experts collaboratively answering a question.
Each one verbosely explains their thought process in real-time, considering the
prior explanations of others and openly acknowledging mistakes. At each step,
whenever possible, each expert refines and builds upon the thoughts of others,
acknowledging their contributions. They continue until there is a definitive
answer to the question. For clarity, your entire response should be in a
markdown table. The question is...
```

### 3. Tree of thoughts experts

```txt
Imagine three highly intelligent experts working together to answer a question.
They will follow a tree of thoughts approach, where each expert shares their
thought process step by step. They will consider the input from others, refine
their thoughts, and build upon the group's collective knowledge. If an expert
realizes their thought is incorrect, they will acknowledge it and withdraw from
the discussion. Continue this process until a definitive answer is reached.
Present the entire response in a markdown table. The question is...
```

### 4. Iterative refinement

```txt
Three experts with exceptional logical thinking skills are collaboratively
answering a question using a tree of thoughts method. Each expert will share
their thought process in detail, taking into account the previous thoughts of
others and admitting any errors. They will iteratively refine and expand upon
each other's ideas, giving credit where it's due. The process continues until
a conclusive answer is found. Organize the entire response in a markdown table
format. The question is...
```

## Roadmap

- [x] Breadth-first search with a beam
- [x] Depth-first search with backtracking and pruning
- [x] Value and vote evaluation
- [ ] Monte Carlo tree search
- [ ] Visualize a search tree from `result.to_dict()`

## Acknowledgements

Thanks to the authors of the paper for sharing this work with the world:

- Shunyu Yao, Princeton University
- Dian Yu, Google DeepMind
- Jeffrey Zhao, Google DeepMind
- Izhak Shafran, Google DeepMind
- Thomas L. Griffiths, Princeton University
- Yuan Cao, Google DeepMind
- Karthik Narasimhan, Princeton University

And thanks to Phil Wang ([lucidrains](https://github.com/lucidrains)) for inspiring me to devote myself to open source AI research.

## Citation

```bibtex
@misc{yao2023tree,
    title         = {Tree of Thoughts: Deliberate Problem Solving with Large Language Models},
    author        = {Shunyu Yao and Dian Yu and Jeffrey Zhao and Izhak Shafran and Thomas L. Griffiths and Yuan Cao and Karthik Narasimhan},
    year          = {2023},
    eprint        = {2305.10601},
    archivePrefix = {arXiv},
    primaryClass  = {cs.CL}
}
```

## License

[Apache 2.0](LICENSE)
