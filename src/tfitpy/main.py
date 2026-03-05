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
    row_dict = row.to_dict()
    row_dict["sources"] = row_dict["sources"].split(";")
    row_pairs = generate_tf_pairs(row_dict["sources"])
    additional_data = {}

    for method_name in methods:
        method_config = METHODS[method_name]
        func = method_config['func']
        resp_type = method_config['type']
        cols = method_config['cols']  # the final column names

        try:
            result = func(datasets=cache, pairs=row_pairs, **options, **row_dict)

            if resp_type == "df_column":
                # single score — cols has exactly one entry
                row_dict[cols[0]] = round(result[0], 5)

            elif resp_type == "df_columns":
                # flat dict returned — keys map 1:1 to cols in order
                for col, val in zip(cols, result.values()):
                    row_dict[col] = round(val, 5) if isinstance(val, float) else val

            elif resp_type == "json":
                additional_data[method_name] = result

        except Exception as e:
            print(f"Error in {method_name} for row {row.name}: {e}")
            for col in cols:
                row_dict[col] = float('nan')

    additional_data["sources"] = ';'.join(row_dict["sources"])
    additional_data["target"] = row_dict["target"]
    row_dict["sources"] = ';'.join(row_dict["sources"])

    return pd.Series(row_dict), additional_data

def compute_indices(df, methods=None, new_methods_only=True, data_path=None, options={}):
    if df is None:
        raise ValueError("No dataframe provided")
    if data_path is None:
        raise ValueError("data_path is required")
    if methods is None:
        methods = list(METHODS.keys())

    required_cols = ['sources', 'target']
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"DataFrame missing required columns: {missing_cols}")

    # Filter out methods whose output columns are already in the dataframe
    if new_methods_only:
        methods = [
            m for m in methods
            if not any(col in df.columns for col in METHODS[m]['cols'])
        ]
        if not methods:
            print("All requested columns already present in dataframe.")
            return df, {}

    print(f"Computing {len(methods)} method(s) on {len(df)} rows")
    #print(df.columns)
    #print(methods)
    cache = load_cache(methods=methods, data_path=data_path)
    print("Working on it...")

    results = []
    additional_data = {}
    for idx, row in df.iterrows():
        result, add_data = _compute_row_indices(row, methods, cache, options)
        results.append(result)
        cluster_uid = row['cluster_uid']
        additional_data[cluster_uid] = add_data

    print(f"Done. Added columns for {len(methods)} method(s).")
    return pd.DataFrame(results), additional_data