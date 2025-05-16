import pandas as pd
import re
from typing import Optional

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