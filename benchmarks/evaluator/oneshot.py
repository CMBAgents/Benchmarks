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

class Evaluator:
    def __init__(self,
                 workdir: str = None,
                 agent: str = 'engineer',
                 model: str = 'gpt-4o',
                 trials: int = 10,
                 rerun: bool = False,
                 verbose: bool = True):
        """
        Initializes the Evaluator class.
        Parameters:
        workdir (str): Directory where the evaluation results will be stored.
        agent (str): The type of agent to use for evaluation, default is 'engineer'.
        model (str): The model to use for evaluation, default is 'gpt-4o'.
        trials (int): Number of trials to run for evaluation, default is 10.
        rerun (bool): If True, rerun the evaluation even if results already exist, default is False.
        verbose (bool): If True, enable verbose logging, default is True.
        """
        self.logger = Logger(self.__class__.__name__, verbose=verbose)
        if workdir is None:
            self.logger.log("No Working Directory provided, using /tmp", level='warning')
            self.workdir = '/tmp' 
        else:
            self.workdir = workdir
            os.makedirs(self.workdir, exist_ok=True)

        self.agent = agent
        self.model = model
        self.trials = trials
        self.rerun = rerun

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
        wdir = os.path.join(self.workdir, f'prompt_{self.agent}{self.model}_{idx}/trial_{trial}')
        os.makedirs(wdir, exist_ok=True)
        return wdir
    

    def ___prompt_cambcontext___(self, idx: int,
                                 trial: int = 0,
                                 *args, **kwargs
                                 ):
        """
        Run the evaluation for a single prompt using cambcontext.
        Parameters:
        idx (int): Index of the prompt to evaluate.
        trial (int): Trial number for the evaluation.
        """
        wdir = self.prompt_dir(idx, trial)
        prompt = self.df.prompt.iloc[idx]
        results = cmbagent.cambcontext(
                prompt,
                max_rounds=50,
                agent=self.agent,
                work_dir=wdir,
                *args,
                **kwargs
            )
        return results
    
    def __prompt_general__(self, idx: int,
                 trial: int = 0,
                 *args, **kwargs):
        """
        Run the evaluation for a single prompt.
        Parameters:
        idx (int): Index of the prompt to evaluate.
        trial (int): Trial number for the evaluation.
        """
        wdir = self.prompt_dir(idx, trial)
        prompt = self.df.prompt.iloc[idx]
        results = cmbagent.one_shot(
                prompt,
                max_rounds=50,
                agent=self.agent,
                engineer_model= self.model,
                work_dir=wdir,
            )
        return results
    
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
            if self.agent == 'camb_context':
                results = self.___prompt_cambcontext___(idx, trial, *args, **kwargs)['final_context']
            else:
                results = self.__prompt_general__(idx, trial, *args, **kwargs)['final_context']
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
    
    def success_prompt(self, idx: int,special_case: bool = True):
        """
        Calculate the success rate for a single prompt.
        Parameters:
        idx (int): Index of the prompt to evaluate.
        """
        if special_case:
            warnings.warn(
                "The 'special_case=True' option will be removed in a future version.",
                FutureWarning,
                stacklevel=2
            )
        s = []
        if special_case:
            results = np.arange(self.trials)
        else:
            results = self.run_prompt_trials(idx)
        for result in results:
            if special_case:
                resultfile = os.path.join(self.prompt_dir(idx,result), 'data', 'result.csv')
            else:
                resultfile = os.path.join(result['work_dir'],
                                    result['database_path'],
                                    'result.csv')
            if not os.path.isfile(resultfile):
                s.append(0)
                continue
            else:
                x_llm, y_llm = np.loadtxt(resultfile, delimiter=',', skiprows=1).T
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

    
            




        
    




