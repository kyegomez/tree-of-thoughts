import re
from typing import Any, Callable, Optional, Tuple, Union

from langchain.llms import OpenAI
from langchain_experimental.tot.checker import ToTChecker
from langchain_experimental.tot.thought import ThoughtValidity


class LangchainTOT:

    def __init__(self,
                 problem_description: Optional[str] = None,
                 checker_class: Optional[Any] = None):
        self.llm = OpenAI(temperature=1, max_tokens=512, model="text-davinci-003")
        self.problem_description = problem_description
        self.checker_class = checker_class if checker_class else ToTChecker
        self.thoughts = []

    def set_problem_description(self, problem_description: str):
        self.problem_description = problem_description

    def set_checker_class(self, checker_class: Any):
        self.checker_class = checker_class

    def add_thought(self, thought: str):
        self.thoughts.append(thought)

    def check_thoughts(self) -> ThoughtValidity:
        if not self.thoughts:
            raise ValueError("No thoughts have been added.")
        if not self.problem_description:
            raise ValueError("Problem description is not set.")
        checker = self.checker_class()
        return checker.evaluate(self.problem_description, tuple(self.thoughts))


class MyChecker(ToTChecker):
    def __init__(self, validate_fn: Callable[[str, Tuple[str, ...]], ThoughtValidity]):
        self.validate_fn = validate_fn

    def evaluate(self, problem_description: str, thoughts: Tuple[str, ...] = ()) -> ThoughtValidity:
        return self.validate_fn(problem_description, thoughts)


def validate_sudoku(problem_description: str, thoughts: Tuple[str, ...], sudoku_solution: str) -> ThoughtValidity:
    last_thought = thoughts[-1]
    clean_solution = last_thought.replace(" ", "").replace('"', "")
    regex_solution = clean_solution.replace("*", ".").replace("|", "\\|")
    if sudoku_solution in clean_solution:
        return ThoughtValidity.VALID_FINAL
    elif re.search(regex_solution, sudoku_solution):
        return ThoughtValidity.VALID_INTERMEDIATE
    else:
        return ThoughtValidity.INVALID


sudoku_solution = "3,4,1,2|1,2,3,4|2,1,4,3|4,3,2,1"
my_checker = MyChecker(validate_fn=lambda p, t: validate_sudoku(p, t, sudoku_solution))

problem_description = """
3,*,*,2|1,*,3,*|*,1,*,3|4,*,*,1

- This is a 4x4 Sudoku puzzle.
- The * represents a cell to be filled.
- The | character separates rows.
- At each step, replace one or more * with digits 1-4.
- There must be no duplicate digits in any row, column or 2x2 subgrid.
- Keep the known digits from previous valid thoughts in place.
- Each thought can be a partial or the final solution.
""".strip()

langchain_tot = LangchainTOT(problem_description=problem_description, checker_class=lambda: my_checker)
langchain_tot.add_thought("3,*,*,2|1,*,3,*|*,1,*,3|4,*,*,1")
print(langchain_tot.check_thoughts())
