import pandas as pd

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