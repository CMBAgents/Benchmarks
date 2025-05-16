import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Union, List
import re

def read_parquet_file(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Read a single Parquet file into a pandas DataFrame.
    
    Args:
        file_path (Union[str, Path]): Path to the Parquet file
        
    Returns:
        pd.DataFrame: DataFrame containing the Parquet data
    """
    return pd.read_parquet(file_path)

def read_parquet_directory(directory_path: Union[str, Path]) -> pd.DataFrame:
    """
    Read all Parquet files in a directory into a single pandas DataFrame.
    
    Args:
        directory_path (Union[str, Path]): Path to the directory containing Parquet files
        
    Returns:
        pd.DataFrame: DataFrame containing all Parquet data concatenated
    """
    directory = Path(directory_path)
    parquet_files = list(directory.glob('*.parquet'))
    
    if not parquet_files:
        raise ValueError(f"No Parquet files found in {directory_path}")
    
    # Read and concatenate all Parquet files
    dfs = [pd.read_parquet(file) for file in parquet_files]
    return pd.concat(dfs, ignore_index=True)

def read_parquet(path: Union[str, Path]) -> pd.DataFrame:
    """
    Read Parquet data from either a file or directory into a pandas DataFrame.
    
    Args:
        path (Union[str, Path]): Path to either a Parquet file or directory containing Parquet files
        
    Returns:
        pd.DataFrame: DataFrame containing the Parquet data
    """
    path = Path(path)
    
    if path.is_file():
        return read_parquet_file(path)
    elif path.is_dir():
        return read_parquet_directory(path)
    else:
        raise ValueError(f"Path {path} does not exist")


def extract_file_path(doc_str: str) -> str:
    """
    Extracts the file path from a formatted documentation string.

    Parameters
    ----------
    doc_str : str
        The documentation string containing a 'File path:' line.

    Returns
    -------
    str
        The extracted file path, or an empty string if not found.
    """
    match = re.search(r'^File path:\s*(.+)', doc_str, re.MULTILINE)
    return match.group(1) if match else ""

# if __name__ == "__main__":
#     # Example usage
#     try:
#         # Example with a single file
#         # df = read_parquet("path/to/your/file.parquet")
        
#         # Example with a directory
#         # df = read_parquet("path/to/your/directory")
        
#         print("Please uncomment and modify the example usage with your actual file/directory path")
#     except Exception as e:
#         print(f"Error: {e}")
