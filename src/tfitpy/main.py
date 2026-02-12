from itertools import combinations, product
import pandas as pd
from joblib import Parallel, delayed
import pandas as pd
import numpy as np


DEFAULT_INDICES = []

DEFAULT_INDEX_OPTIONS= {

}

def load_required_data(selected_indices):
   """
   This method loads all the common data required to compute indices and stores in the the global COMMON_STORE.
   """
  

def indices_for_item(row, indices=None, index_options=None, data_path=None):
    """
    Compute indices for an individual row
    
    :param chunk: Description
    :param indices: Description
    :param index_options: Description
    :param data_path: Description
    """
    row["new_col1"] = "hi"  # Use params as needed, e.g., load data from data_path
    row["new_col2"] = "hellooo"
    row["new_col3"] = 1
    return row


def parallel_apply(df, func, n_jobs=-1, **kwargs):
    """
   To run computations in parallel 
    
    :param df: Input dataframe
    :param func: method that computes indices of individual case
    :param n_jobs: Number of jobs to run in parallel
    """
    splits = np.array_split(df.index, max(1, n_jobs)) 
    results = Parallel(n_jobs=n_jobs)(delayed(func)(df.loc[split_idx], **kwargs) for split_idx in splits)
    return pd.concat(results, ignore_index=False)



##### The main method

def compute_indices(clusters=None, n_jobs=1, indices=None, index_options=None, sources_separator=";", data_path=None):
  """
  Takes a dataframe of clusters with cols 'sources', 'target'. Sources must be separated with the separator specified.

  """
  if clusters is None:
    raise ValueError("No dataframe provided")
  
  if data_path is None:
     raise ValueError("Provide a path where datasets are stored/ will be downloaded")

  if indices is None:
    indices = DEFAULT_INDICES
  if index_options is None:
    index_options = DEFAULT_INDEX_OPTIONS

  if 'sources' not in clusters.columns:
    raise ValueError("Dataframe does not have the sources col")
  
  if 'target' not in clusters.columns:
    raise ValueError("Dataframe does not have target column")
  
  dataset_cache = load_required_data()


  df = parallel_apply(clusters, indices_for_item, n_jobs=-1,dataset_cache=dataset_cache)
  return df
