from benchmarks import Logger
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from benchmarks.utils import run_function
import cmbagent
import pickle as pl
import matplotlib.pyplot as plt
import warnings
import hashlib


class Evaluator:
    def __init__(self,
                workdir: str = None,
                trials: int = 10,
                max_rounds_control: int = 500,
                n_plan_reviews: int = 1,
                max_n_attempts: int = 3,
                max_plan_steps: int = 4,
                default_llm_model: str = "gpt-4.1-2025-04-14",
                engineer_model: str = "gemini-2.5-pro-preview-03-25",
                camb_context_model: str = "gemini-2.5-pro-preview-03-25",
                researcher_model: str = "gpt-4.1-2025-04-14",
                plan_reviewer_model: str = "claude-3-7-sonnet-20250219",
                plan_instructions: str=r"""
Use camb_context agent to start and then engineer agent for the whole analysis. 
The plan must have 3 steps or more. """):
        
        """
        Initializes the Evaluator class.
        Parameters:
        workdir (str): Directory where the evaluation results will be stored.
        max_rounds_control (int): Maximum number of control rounds.
        n_plan_reviews (int): Number of plan reviews.
        max_n_attempts (int): Maximum number of attempts for evaluation.
        max_plan_steps (int): Maximum number of steps in the plan.
        default_llm_model (str): Default LLM model to use.
        engineer_model (str): Model used for engineering tasks.
        camb_context_model (str): Model used for context carrying over.
        researcher_model (str): Model used for research tasks.
        plan_reviewer_model (str): Model used for reviewing plans.
        plan_instructions (str): Instructions for the planning process.
        """
        self.logger = Logger(self.__class__.__name__, verbose=True)
        if workdir is None:
            self.logger.log("No Working Directory provided, using /tmp", level='warning')
            self.workdir = '/tmp'
        else:
            self.workdir = workdir
            os.makedirs(self.workdir, exist_ok=True)
        self.trials = trials
        self.max_rounds_control = max_rounds_control
        self.n_plan_reviews = n_plan_reviews
        self.max_n_attempts = max_n_attempts
        self.max_plan_steps = max_plan_steps
        self.default_llm_model = default_llm_model
        self.engineer_model = engineer_model
        self.camb_context_model = camb_context_model
        self.researcher_model = researcher_model
        self.plan_reviewer_model = plan_reviewer_model
        self.plan_instructions = plan_instructions
        

    def __call__(self,df: pd.DataFrame,
                 ref_dir: str = None,
                 rerun_ref: bool = False):
        """
        Intialize the dataframe with the columns required for evaluation.
        Parameters:
        df (pd.DataFrame): DataFrame containing the evaluation data.
        """
        self.df = df.copy()
        self.logger.log(f"Dataframe contains ({len(self.df)}) prompts", level='info')
        assert 'prompt' in self.df.columns, "DataFrame must contain a 'prompt' column"
        assert 'reference_code' in self.df.columns, "DataFrame must contain a 'reference_code' column"

        if ref_dir is None:
            self.ref_dir = os.path.join(self.workdir, 'reference')
        else:
            self.ref_dir = ref_dir
        os.makedirs(self.ref_dir, exist_ok=True)
        self.logger.log(f"Reference directory set to {self.ref_dir}", level='info')

        for i in tqdm(range(len(self.df)), desc="creating reference answers for prompts"):
            ref_code = self.df.reference_code.iloc[i]
            fname = f'prompt_{i}.csv'
            if os.path.isfile(os.path.join(self.ref_dir, fname)) and not rerun_ref:
                continue
            else:
                x,y = run_function(ref_code)
                np.savetxt(os.path.join(self.ref_dir, fname),np.array([x,y]).T, delimiter=',', header='x,y', comments='')
    
    def prompt_dir(self, idx: int,
                   trial: int = 0):
        """
        Run the evaluation for a single prompt.
        Parameters:
        idx (int): Index of the prompt to evaluate.
        """
        fname = f"{self.max_rounds_control}{self.n_plan_reviews}{self.max_n_attempts}{self.max_plan_steps}{self.default_llm_model}{self.engineer_model}{self.camb_context_model}{self.researcher_model}{self.plan_reviewer_model}"
        short_fname = hashlib.md5(fname.encode()).hexdigest()[:8]
        wdir = os.path.join(self.workdir, f'prompt_{short_fname}_{idx}', f'trial_{trial}')
        os.makedirs(wdir, exist_ok=True)
        return wdir
    
    def run_prompt(self, idx: int,
                 trial: int = 0,
                 *args, **kwargs):
        """
        Run the evaluation for a single prompt.
        Parameters:
        idx (int): Index of the prompt to evaluate.
        trial (int): Trial number for the evaluation.
        """
        wdir = self.prompt_dir(idx, trial)
        fname = os.path.join(wdir, 'results.pkl')
        if os.path.isfile(fname) and not self.rerun:
            self.logger.log(f"Results for prompt {idx}, trial {trial} already exist, skipping", level='info')
            results = pl.load(open(fname, 'rb'))
        else:
            prompt = self.df.prompt.iloc[idx]
            results = cmbagent.planning_and_control_context_carryover(prompt,
                                max_rounds_control = self.max_rounds_control,
                                n_plan_reviews = self.n_plan_reviews,
                                max_n_attempts = self.max_n_attempts,
                                max_plan_steps = self.max_plan_steps,
                                default_llm_model = self.default_llm_model,
                                engineer_model = self.engineer_model,
                                camb_context_model = self.camb_context_model,
                                researcher_model = self.researcher_model,
                                plan_reviewer_model = self.plan_reviewer_model,
                                plan_instructions = self.plan_instructions,
                                work_dir = wdir,)['final_context']
            pl.dump(results, open(fname, 'wb'))
        return results
    
    def run_prompt_trials(self, idx: int,
                            *args, **kwargs):
        """
        Run multiple trials for a single prompt.
        Parameters:
        idx (int): Index of the prompt to evaluate.
        """
        results = []
        for trial in range(self.trials):
            result = self.run_prompt(idx, trial, *args, **kwargs)
            results.append(result)
        return results
    
    def run_all(self, *args, **kwargs):
        """
        Run the evaluation for all prompts in the DataFrame.
        """
        self.logger.log(f"Running evaluation for {len(self.df)} prompts", level='info')
        results = []
        for idx in tqdm(range(len(self.df)), desc="Running all prompts"):
            result = self.run_prompt_trials(idx, *args, **kwargs)
            results.append(result)
        return results
    
    def success_prompt(self, idx: int):
        """
        Calculate the success rate for a single prompt.
        Parameters:
        idx (int): Index of the prompt to evaluate.
        """
        s = []
        results = self.run_prompt_trials(idx)
        for result in results:
            resultfile = os.path.join(result['work_dir'],
                                    result['database_path'],
                                    'result.csv')
            if not os.path.isfile(resultfile):
                s.append(0)
                continue
            else:
                try:
                    x_llm, y_llm = np.loadtxt(resultfile, delimiter=',', skiprows=1).T
                except:
                    s.append(0)
                    continue
            x_ref, y_ref = np.loadtxt(os.path.join(self.ref_dir, f'prompt_{idx}.csv'), delimiter=',', skiprows=1).T
            if len(x_llm) != len(x_ref):
                s.append(0)
                continue
            if np.allclose(x_llm, x_ref) and np.allclose(y_llm, y_ref, rtol=1e-1):
                s.append(1)
            else:
                s.append(0)

            # plt.loglog(x_llm, y_llm, label='LLM')
            # plt.loglog(x_ref, y_ref, label='True')
            # plt.legend()
            # plt.savefig(os.path.join(result['work_dir'], 'cl_plot.png'))
            # plt.close()
        return np.mean(s)
    
    def success_all(self, special_case: bool = True):
        """
        Calculate the success rate for all prompts in the DataFrame.
        Parameters:
        special_case (bool): If True, use the special case for success calculation.
        """
        results = []
        for idx in range(len(self.df)):
            result = self.success_prompt(idx, special_case=special_case)
            results.append(result)
        return results

         