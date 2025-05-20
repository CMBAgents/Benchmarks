import pandas as pd
import re
from typing import Optional
import ast
import textwrap

# --- helper --------------------------------------------------------------
def make_prompt(row: pd.Series) -> str:
    """
    Build a single markdown string that shows the problem first
    and then the code-answer requirements.

    Parameters
    ----------
    row : pd.Series
        A row that has `problem` and `code_answer_requirements` fields.

    Returns
    -------
    str
        A markdown-friendly prompt string.
    """
    return (
        "### Problem\n"
        f"{row.problem.strip()}\n\n"
        "### Requirements\n"
        f"{row.code_answer_requirements.strip()}"
    )

def extract_python_definition(env_str: str) -> Optional[str]:
    """
    Extracts the Python code between \\begin{python} and \\end{python} markers.

    Parameters
    ----------
    env_str : str
        A string that includes a \\begin{python}…\\end{python} block.

    Returns
    -------
    Optional[str]
        The code inside the block, or None if no such block is found.
    """
    pattern = r"\\begin\{python\}([\s\S]*?)\\end\{python\}"
    m = re.search(pattern, env_str)
    if not m:
        return None
    # Strip any leading/trailing whitespace
    return m.group(1).strip()






def extract_first_function(summary: str) -> Optional[str]:
    """
    Return the source code (including docstring and body) of the first
    top-level function that appears inside a ```python ... ``` code block
    in a `this_step_execution_summary` string.

    If nothing is found, returns None.
    """
    # 1) Locate the *first* fenced code block that is marked as Python
    match = re.search(r"```python\s*(.*?)```", summary, re.S | re.I)
    if not match:                                    # no python block
        return None
    code_block = textwrap.dedent(match.group(1))     # strip common indent

    # 2) Parse that code so we can discover where the function lives
    try:
        module = ast.parse(code_block)
    except SyntaxError:                              # malformed code
        return None

    # 3) Grab the *first* top-level ast.FunctionDef, if any
    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            # lineno / end_lineno are 1-based and inclusive
            lines = code_block.splitlines()
            func_src = "\n".join(lines[node.lineno - 1 : node.end_lineno])
            return func_src.rstrip()                 # drop trailing newline

    return None
