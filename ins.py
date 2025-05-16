import ast
import inspect
import random
import sys
import asyncio
import pandas as pd                    
import inspect
from inspect_ai import Task, task, eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import exact   
from inspect_ai.solver import solver, TaskState, Generate
from inspect_ai.scorer import scorer, mean, stderr, CORRECT, INCORRECT, Score
from inspect_ai.solver import TaskState
import difflib

@solver
def submission(source: str):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.output.completion = source
        return state
    return solve


@scorer(metrics=[mean()])
def same_io_dynamic(num_tests: int = 50, low: int = -100, high: int = 100):
    """
    Test functions of any arity by:
      1. Parsing `reference_code` to get the correct function name.
      2. Loading that same name from the candidate file.
      3. Running randomized I/O tests with numpy→Python coercion.
    """
    async def score(state: TaskState, target) -> Score:
        cand_src = state.output.completion
        ref_src  = getattr(target, "text", target)


        ref_tree = ast.parse(ref_src)
        ref_defs = [n for n in ref_tree.body if isinstance(n, ast.FunctionDef)]
        if not ref_defs:
            return Score(value=INCORRECT,
                         answer=cand_src,
                         explanation="Reference has no function definition.")
        fn_name = ref_defs[0].name


        def load_named_fn(src: str, name: str):
            ns = {}
            exec(compile(src, "<string>", "exec"), ns)
            if name not in ns or not callable(ns[name]):
                raise ValueError(f"Function {name!r} not found in candidate.")
            return ns[name]

        try:
            cand_fn = load_named_fn(cand_src, fn_name)
            ref_fn  = load_named_fn(ref_src,  fn_name)
        except Exception as e:
            return Score(value=INCORRECT,
                         answer=cand_src,
                         explanation=f"Load error: {e}")


        sig = inspect.signature(ref_fn)
        param_count = len(sig.parameters)

        for _ in range(num_tests):
            args = [random.randint(low, high) for _ in range(param_count)]
            try:
                c_out = cand_fn(*args)
                r_out = ref_fn(*args)
            except Exception as e:
                return Score(value=INCORRECT,
                             answer=cand_src,
                             explanation=f"Runtime error on {args}: {e}")

            c_val = c_out.item() if hasattr(c_out, "item") else c_out
            r_val = r_out.item() if hasattr(r_out, "item") else r_out

            if c_val != r_val:
                return Score(
                    value=INCORRECT,
                    answer=cand_src,
                    explanation=(
                        f"Mismatch for inputs {args}: got {c_val} "
                        f"({type(c_val)}), expected {r_val} ({type(r_val)})"
                    )
                )

        return Score(value=CORRECT, answer=cand_src)

    return score


@scorer(metrics=[mean()])
def content_similarity():
    """
    Compute content similarity between the candidate and reference functions
    by comparing their AST dumps via difflib.
    Returns a ratio in [0,1] indicating structural similarity.
    """
    async def score(state: TaskState, target) -> Score:

        cand_src = state.output.completion
        ref_src = getattr(target, "text", target)


        try:
            cand_ast = ast.parse(cand_src)
            ref_ast = ast.parse(ref_src)
        except SyntaxError as e:
            return Score(value=0.0, answer=cand_src, explanation=f"Syntax error during parse: {e}")

        cand_dump = ast.dump(cand_ast, annotate_fields=False)
        ref_dump = ast.dump(ref_ast, annotate_fields=False)


        ratio = difflib.SequenceMatcher(None, cand_dump, ref_dump).ratio()


        return Score(value=ratio, answer=cand_src, explanation=f"AST similarity ratio: {ratio:.2f}")

    return score

@task
def code_match_dynamic(reference,candidate):
    return Task(
        dataset=[Sample(
            input="Return Python code solving the task.",
            target=reference
        )],
        solver=[submission(candidate)],
        scorer=same_io_dynamic()
    )
@task
def code_match_with_similarity(reference,candidate): 
    return Task(
        dataset=[Sample(input="Return Python code solving the task.", target=reference)],
        solver=[submission(candidate)],  
        scorer=[same_io_dynamic(),content_similarity()]
    )


def evaluate(reference_code, candidate_code):
    logs = inspect_eval(code_match_with_similarity(reference_code,candidate_code), model="openai/gpt-4.1-2025-04-14")
    l ={}
    for i in range(2):
        l[logs[0].results.scores[i].name] = logs[0].results.scores[i].metrics['mean'].value
    return l
    