import os
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count

from tfitpy.utils import generate_tf_pairs
from tfitpy.datasets import DATASETS
from tfitpy.indices import METHODS

# --- Module-level cache (one per worker process) ---
_worker_cache = {}

def _worker_init(methods: list, data_path: str):
    """
    Called once per worker process at pool startup.
    Builds the dataset cache into the module-level _worker_cache dict.
    Runs in the worker's own address space — no sharing across processes.
    """
    global _worker_cache
    _worker_cache = load_cache(methods=methods, data_path=data_path)


def load_cache(cache=None, methods=None, data_path=None):
    """
    Load datasets into a key-value cache for the specified methods.

    Args:
        cache:     Existing cache dict to update. Creates a new one if None.
        methods:   Method names whose required datasets should be loaded.
                   Defaults to all known methods.
        data_path: Directory where dataset files are stored.

    Returns:
        Cache dict mapping dataset name → loaded dataset object.
    """
    if cache is None:
        cache = {}
    if methods is None:
        methods = list(METHODS.keys())
    if data_path is None:
        raise ValueError("data_path is required")

    data_path = Path(data_path)

    invalid = set(methods) - set(METHODS.keys())
    if invalid:
        raise ValueError(f"Unknown methods: {invalid}")

    needed = {ds for m in methods for ds in METHODS[m]["datasets"]}
    to_load = [ds for ds in needed if ds not in cache]

    if not to_load:
        #print("All required datasets already in cache.")
        return cache

    #print(f"[pid {os.getpid()}] Loading {len(to_load)} dataset(s): {to_load}")
    for ds_name in to_load:
        if ds_name not in DATASETS:
            raise ValueError(f"Dataset '{ds_name}' not found in registry.")
        try:
            cache[ds_name] = DATASETS[ds_name]["load"](data_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset '{ds_name}': {e}") from e

    return cache


def _compute_row_indices(row, methods, cache, options):
    """
    Compute all requested method indices for a single DataFrame row.

    Returns:
        (pd.Series, dict)  — updated row series and any json-type side data.
    """
    row_dict = row.to_dict()
    row_dict["sources"] = row_dict["sources"].split(";")
    row_pairs = generate_tf_pairs(row_dict["sources"])
    additional_data = {}

    for method_name in methods:
        method_config = METHODS[method_name]
        func = method_config["func"]
        resp_type = method_config["type"]
        cols = method_config["cols"]

        try:
            result = func(datasets=cache, pairs=row_pairs, **options, **row_dict)

            if resp_type == "df_column":
                row_dict[cols[0]] = round(result[0], 5)

            elif resp_type == "df_columns":
                for col in cols:
                    val = result[col]
                    row_dict[col] = round(val, 5) if isinstance(val, float) else val

            elif resp_type == "json":
                additional_data[method_name] = result

        except Exception as e:
            # print(f"Error in {method_name} for row {row.name}: {e}")
            for col in cols:
                row_dict[col] = float("nan")

    sources_str = ";".join(row_dict["sources"])
    additional_data["sources"] = sources_str
    additional_data["target"] = row_dict["target"]
    row_dict["sources"] = sources_str

    return pd.Series(row_dict), additional_data


def _process_chunk(args):
    """
    Worker entry point — processes a chunk of rows using the pre-built
    module-level cache (_worker_cache).  No cache is passed in; it lives
    in the worker's own memory from _worker_init.

    Args:
        args: (chunk_df, methods, options)

    Returns:
        (list[pd.Series], dict)  — row results and aggregated additional data.
    """
    chunk_df, methods, options = args
    results = []
    additional_data = {}

    for _, row in chunk_df.iterrows():
        series, add_data = _compute_row_indices(row, methods, _worker_cache, options)
        results.append(series)
        cluster_uid = row["cluster_uid"]
        additional_data[cluster_uid] = add_data

    return results, additional_data


def compute_indices(
    df,
    methods=None,
    new_methods_only=True,
    data_path=None,
    options={},
    n_jobs=None
):
    """
    Compute index columns for every row of *df* in parallel.

    Each worker process builds its own independent dataset cache on startup
    (via the pool initializer), so datasets are never pickled or shared across
    process boundaries.  The DataFrame is split into one chunk per worker and
    distributed via starmap.

    Args:
        df:              Input DataFrame. Must contain 'sources', 'target',
                         and 'cluster_uid' columns.
        methods:         Method names to run. Defaults to all known methods.
        new_methods_only: Skip methods whose output columns already exist in df.
        data_path:       Path to dataset files (required).
        options:         Extra keyword arguments forwarded to each method func.
        n_jobs:          Number of worker processes. Defaults to cpu_count().

    Returns:
        (pd.DataFrame, dict)  — augmented DataFrame and merged additional_data.
    """
    if df is None:
        raise ValueError("No dataframe provided.")
    if data_path is None:
        raise ValueError("data_path is required.")
    if methods is None:
        methods = list(METHODS.keys())

    for col in ("sources", "target"):
        if col not in df.columns:
            raise ValueError(f"DataFrame missing required column: '{col}'")

    if new_methods_only:
        methods = [
            m for m in methods
            if not any(col in df.columns for col in METHODS[m]["cols"])
        ]
        if not methods:
            # print("All requested columns already present — nothing to compute.")
            return df, {}

    n_workers = min(n_jobs or cpu_count(), len(df))
    # print(f"Computing {len(methods)} method(s) on {len(df)} rows using {n_workers} worker(s).")

    # Split into one chunk per worker; leftover rows go into the last chunk.
    chunks = [chunk for chunk in _split_dataframe(df, n_workers) if not chunk.empty]

    chunk_args = [(chunk, methods, options) for chunk in chunks]

    with Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(methods, str(data_path)),
    ) as pool:
        outcomes = pool.map(_process_chunk, chunk_args)
        pool.close()
        pool.join()

    # Merge results from all workers
    all_rows = []
    merged_additional: dict = {}
    for rows, add_data in outcomes:
        all_rows.extend(rows)
        merged_additional.update(add_data)

    #print(f"Done. Added columns for {len(methods)} method(s).")
    return pd.DataFrame(all_rows), merged_additional


def _split_dataframe(df: pd.DataFrame, n: int):
    """Yield *n* roughly equal sub-DataFrames."""
    size = max(1, len(df) // n)
    for start in range(0, len(df), size):
        yield df.iloc[start: start + size]