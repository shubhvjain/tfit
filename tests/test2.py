from tfitpy import compute_indices
import pandas as pd
from pathlib import Path
import os

if __name__ == "__main__":
  csv_path = Path(__file__).resolve().parent / "cluster_groups.csv"
  csv_path_results = Path(__file__).resolve().parent / "cluster_results.csv"
  df = pd.read_csv(csv_path)
  print(df)
  data_path = Path(os.path.expandvars("$HOME/projects/bio-datasets"))
  results = compute_indices(df,data_path=data_path)
  print(results)
  results.to_csv(csv_path_results)

