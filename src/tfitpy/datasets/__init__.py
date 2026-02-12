from tfitpy.datasets.grn import GRN_DATASETS
from tfitpy.datasets.ppi import PPI_DATASETS
from tfitpy.datasets.gene_names import GENE_DATASETS


DATASETS = {
  ** GENE_DATASETS,
  ** GRN_DATASETS,
  ** PPI_DATASETS
}
 
FIRST_ORDER  =  ["biomart"]