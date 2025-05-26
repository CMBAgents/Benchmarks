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
import numpy as np
import ast

def run_single_function(code: str):
    """
    Executes Python code defining one function, and returns the output of that function.
    The function must take no arguments.
    """
    # Step 1: Parse code and find the function name
    tree = ast.parse(code)
    func_name = next(node.name for node in tree.body if isinstance(node, ast.FunctionDef))

    # Step 2: Create one shared namespace for both imports and function
    shared_namespace = {}
    exec(code, shared_namespace)

    # Step 3: Call the function
    output = shared_namespace[func_name]()  # Call with no arguments
    return output

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
@scorer(metrics=[mean()])
def no_input_array_io(rtol: float = 1e-6, atol: float = 1e-8):
    """
    Test zero-argument functions that return arrays (or tuples of arrays) by:
      1. Parsing `reference_code` for the function name.
      2. Loading that same name from the candidate file.
      3. Calling each with no arguments.
      4. Verifying output shapes (and optionally values) match.
    """
    async def score(state: TaskState, target) -> Score:
        cand_src = state.output.completion
        ref_src = getattr(target, "text", target)

        # Find the first function in the reference
        ref_tree = ast.parse(ref_src)
        ref_defs = [n for n in ref_tree.body if isinstance(n, ast.FunctionDef)]
        if not ref_defs:
            return Score(
                value=INCORRECT,
                answer=cand_src,
                explanation="Reference has no function definition."
            )
        fn_name = ref_defs[0].name

        # Helper to exec & retrieve named function
        def load_fn(src: str, name: str):
            ns = {}
            exec(compile(src, "<string>", "exec"), ns)
            if name not in ns or not callable(ns[name]):
                raise ValueError(f"Function {name!r} not found.")
            return ns[name]

        try:
            cand_fn = load_fn(cand_src, fn_name)
            ref_fn = load_fn(ref_src, fn_name)
        except Exception as e:
            return Score(
                value=INCORRECT,
                answer=cand_src,
                explanation=f"Load error: {e}"
            )

        # Call both functions with no args
        try:
            c_out = cand_fn()
            r_out = ref_fn()
        except Exception as e:
            return Score(
                value=INCORRECT,
                answer=cand_src,
                explanation=f"Runtime error when calling {fn_name}(): {e}"
            )

        # Normalize to tuples
        c_tup = c_out if isinstance(c_out, tuple) else (c_out,)
        r_tup = r_out if isinstance(r_out, tuple) else (r_out,)

        if len(c_tup) != len(r_tup):
            return Score(
                value=INCORRECT,
                answer=cand_src,
                explanation=(
                    f"Return-value count mismatch: got {len(c_tup)}, "
                    f"expected {len(r_tup)}."
                )
            )

        # Compare each returned element
        for i, (c_val, r_val) in enumerate(zip(c_tup, r_tup)):
            if isinstance(c_val, np.ndarray) and isinstance(r_val, np.ndarray):
                if c_val.shape != r_val.shape:
                    return Score(
                        value=INCORRECT,
                        answer=cand_src,
                        explanation=(
                            f"Output #{i} shape mismatch: "
                            f"{c_val.shape} vs {r_val.shape}"
                        )
                    )
                # Optional: check numerical closeness
                # if not np.allclose(c_val, r_val, rtol=rtol, atol=atol):
                #     return Score(
                #         value=INCORRECT,
                #         answer=cand_src,
                #         explanation=(
                #             f"Values differ beyond tolerance in output #{i}."
                #         )
                #     )
            else:
                if c_val != r_val:
                    return Score(
                        value=INCORRECT,
                        answer=cand_src,
                        explanation=(
                            f"Output #{i} mismatch: got {c_val} "
                            f"({type(c_val)}), expected {r_val} ({type(r_val)})"
                        )
                    )

        return Score(value=CORRECT, answer=cand_src)

    return score
@task
def camb_with_similarity(reference,candidate): 
    return Task(
        dataset=[Sample(input="Return Python code solving the task.", target=reference)],
        solver=[submission(candidate)],  
        scorer=[no_input_array_io(),content_similarity()]
    )
                
def evaluate(reference_code, candidate_code):
    logs = inspect_eval(code_match_with_similarity(reference_code,candidate_code), model="openai/gpt-4.1-2025-04-14")
    l ={}
    for i in range(2):
        l[logs[0].results.scores[i].name] = logs[0].results.scores[i].metrics['mean'].value
    return l

def evaluate_camb(reference_code, candidate_code):
    logs = inspect_eval(camb_with_similarity(reference_code,candidate_code), model="openai/gpt-4.1-2025-04-14")
    l ={}
    for i in range(2):
        l[logs[0].results.scores[i].name] = logs[0].results.scores[i].metrics['mean'].value
    return l

