import pandas as pd
from joblib import Parallel, delayed
from pathlib import Path
from tfitpy.utils import generate_tf_pairs

from tfitpy.datasets import DATASETS
from tfitpy.indices import METHODS 


def load_cache(cache=None, methods=None, data_path=None):
    """
    Load datasets into a key value cache variable for specified methods.
    
    This function can either create a new cache or add to an existing one.
    It loads only the datasets required by the specified methods.
    
    Args:
        cache: Existing cache dict to update (optional). If None, creates new cache.
        methods: List of method names that will be used. Loads their required datasets.
        data_path: Path where datasets are stored (from setup_datasets)
    
    Returns:
        Cache dict with loaded datasets
        
    Examples:
        # Create new cache
        cache = load_datasets(methods=['m1', 'm2'], data_path='./data')
        
        # Add more datasets to existing cache
        load_datasets(cache, methods=['m3'], data_path='./data')
    """
    # Create new cache if none provided
    if cache is None:
        cache = {}
    
    if methods is None:
        methods = list(METHODS.keys())
    
    if data_path is None:
        raise ValueError("data_path parameter is required")
    
    data_path = Path(data_path)
    
    # Validate methods exist
    invalid_methods = set(methods) - set(METHODS.keys())
    if invalid_methods:
        raise ValueError(f"Unknown methods: {invalid_methods}")
    
    # Determine which datasets are needed
    datasets_needed = set()
    for method_name in methods:
        method_config = METHODS[method_name]
        datasets_needed.update(method_config['datasets'])
    
    # Filter out datasets already in cache
    datasets_to_load = [ds for ds in datasets_needed if ds not in cache]
    
    if len(datasets_to_load) == 0:
        print("All required datasets already loaded in cache")
        return cache
    
    print(f"Loading {len(datasets_to_load)} dataset(s): {list(datasets_to_load)}")
    print("=" * 60)
    
    # Load each dataset
    for ds_name in datasets_to_load:
        if ds_name not in DATASETS:
            raise ValueError(f"Dataset '{ds_name}' not found in registry")
        
        ds_config = DATASETS[ds_name]
        print(f"Loading {ds_name}...")
        
        try:
            cache[ds_name] = ds_config['load'](data_path)
            print(f"{ds_name} loaded")
        except Exception as e:
            print(f"{ds_name} failed to load: {e}")
            raise
    
    print("=" * 60)
    print(f"Cache now contains {len(cache)} dataset(s): {list(cache.keys())}")
    
    return cache


def _compute_row_indices(row, methods, cache, options):
    """
    Compute indices for a single row.
    Cache is shared read-only across all workers.
    
    Args:
        row: Single row (Series) from dataframe
        methods: List of method names to compute
        cache: Pre-loaded datasets (pickeable, read-only)
    
    Returns:
        Row with new columns added
    """
    row_dict = row.to_dict()
    row_dict["sources"] = row_dict["sources"].split(";")
    row_pairs = generate_tf_pairs(row_dict["sources"])
    additional_data = {}
    for method_name in methods:
        method_config = METHODS[method_name]
        func = method_config['func']
        
        try:
            # Call method with row data and cache
            result = func(datasets=cache,pairs=row_pairs,**options,**row_dict)
            if type(result)== tuple:
                row_dict[method_name] = round(result[0],5)
            elif type(result) == dict:
                additional_data[method_name] = result
                #row_dict[method_name] = result
        except Exception as e:
            print(e)
            print(f"Error in {method_name} for row {row.name}: {e}")
            row_dict[method_name] = None
            # raise e
            pass
    additional_data["sources"] = ';'.join(row_dict["sources"])
    additional_data["target"] = row_dict["target"]

    row_dict["sources"] = ';'.join(row_dict["sources"])
    print(additional_data,row_dict)
    return pd.Series(row_dict),additional_data

def compute_indices(df, methods=None, data_path=None, options={}):
    """
    Compute multiple indices/methods on a dataframe in parallel.
    
    Args:
        df: Input dataframe with 'sources', 'target', etc.
        methods: List of method names to compute (default: all methods)
        data_path: Path where datasets are stored
        n_jobs: Number of parallel jobs (-1 = all cores)
    
    Returns:
        Dictionary with cluster names as keys and their data as values.
        Each value contains genes, target, and computed indices.
    """
    if df is None:
        raise ValueError("No dataframe provided")
    
    if data_path is None:
        raise ValueError("data_path is required")
    
    if methods is None:
        methods = list(METHODS.keys())
    
    # Validate required columns
    required_cols = ['sources', 'target']
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"DataFrame missing required columns: {missing_cols}")
    
    print(f"Computing {len(methods)} method(s) on {len(df)}")
    
    # Load cache ONCE - shared in memory across threads
    cache = load_cache(methods=methods, data_path=data_path)
    print("working on it...")
    # Use threading backend to share cache in memory
    # results = Parallel(n_jobs=n_jobs,backend='threading')(
    #     delayed(_compute_row_indices)(row, methods, cache, options)
    #     for idx, row in df.iterrows()
    # )
    results = []
    additional_data = {}
    for idx, row in df.iterrows():
        result,add_data = _compute_row_indices(row, methods, cache, options)
        results.append(result)
        cluster_uid = row['cluster_uid'] 
        additional_data[cluster_uid] = add_data
    
    # Convert directly to dict
    # result_dict = {}
    # for series_result in results:
    #     row_dict = series_result.to_dict()
    #     cluster_uid = row_dict['cluster_uid']  
    #     result_dict[cluster_uid] = row_dict

    print(f"Computation complete. Added {len(methods)} column(s)")

    
    print(f"Done")
    return pd.DataFrame(results) , additional_data