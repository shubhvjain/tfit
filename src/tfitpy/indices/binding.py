from pyfaidx import Fasta
from Bio import motifs
from Bio.Seq import Seq
from Bio.motifs.thresholds import ScoreDistribution
import pandas as pd
from pathlib import Path
from pyjaspar import jaspardb

from tfitpy.datasets.binding import load_jaspar, get_jasper_path
from tfitpy.datasets.gene_names import load_gencode, load_genome


PROMOTER_UPSTREAM = 2000
PROMOTER_DOWNSTREAM = 200
# Uniform background — adjust to genome GC content if desired (human ~0.41 GC)
_BACKGROUND = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}


def get_promoter_sequence(
    gene_symbol: str,
    data_path,
    upstream: int = PROMOTER_UPSTREAM,
    downstream: int = PROMOTER_DOWNSTREAM,
    datasets=None,
):
    """
    Fetch the promoter sequence for a single gene symbol.
    Slices [TSS - upstream, TSS + downstream] from the genome FASTA.
    Returns the sequence as a string, or None if gene not found.
    """
    if datasets is None:
        raise ValueError("datasets cache is required.")

    con = datasets['gencode']
    row = pd.read_sql_query(
        "SELECT chromosome, strand, tss FROM mappings WHERE gene_name = ? AND feature = 'gene' LIMIT 1",
        con, params=[gene_symbol]
    )

    if row.empty:
        print(f"Gene not found: {gene_symbol}")
        return None

    chrom = row.iloc[0]['chromosome']
    strand = row.iloc[0]['strand']
    tss = int(row.iloc[0]['tss'])

    start = max(0, tss - upstream)
    end = tss + downstream

    genome = datasets["genome"]

    if chrom not in genome:
        print(f"Chromosome {chrom} not found in FASTA")
        return None

    seq = genome[chrom][start:end].seq

    if strand == '-':
        seq = str(Seq(seq).reverse_complement())

    return seq


def get_cache(data_path):
    cache = {}

    jaspar_db = get_jasper_path(data_path)
    jdb = jaspardb(sqlite_db_path=str(jaspar_db))

    cache["jaspar"] = jdb
    cache["gencode"] = load_gencode(data_path)
    cache["genome"] = load_genome(data_path)
    return cache

def scan_promoter(
    gene_symbol,
    tf_symbols,
    data_path,
    upstream=PROMOTER_UPSTREAM,
    downstream=PROMOTER_DOWNSTREAM,
    pseudocount=0.1,
    fpr=0.001,
    datasets=None,
):
    """
    Scan the promoter of a target gene for binding sites of a set of TFs.

    Args:
        gene_symbol : target gene whose promoter is scanned
        tf_symbols  : list of TF gene symbols (candidate regulators)
        data_path   : root data directory
        upstream    : bp upstream of TSS to include
        downstream  : bp downstream of TSS to include
        pseudocount : added to PFM counts before log-odds scoring
        fpr         : false positive rate per position (default 0.001).
                      The score threshold for each motif is derived from its
                      background score distribution so that the probability of
                      a random sequence exceeding it is <= fpr. This makes
                      thresholds comparable across motifs of different lengths
                      and information contents.
        datasets    : cache dict from get_cache()

    Returns:
        DataFrame with columns: tf_name, motif_id, position, motif_length,
                                 strand, score
        position is always a non-negative forward-strand coordinate from the
        start of the promoter window (0 = upstream edge, upstream value = TSS).
    """
    if isinstance(tf_symbols, str):
        tf_symbols = [tf_symbols]

    sequence = get_promoter_sequence(
        gene_symbol, data_path, upstream, downstream, datasets)
    if sequence is None:
        return pd.DataFrame()

    jdb = datasets["jaspar"]

    motif_list = jdb.fetch_motifs(
        collection='CORE',
        tax_group=['Vertebrates'],
        tf_name=tf_symbols,
        all_versions=False,
    )

    if not motif_list:
        print(f"No JASPAR motifs found for: {tf_symbols}")
        return pd.DataFrame()

    hits = []
    seq = Seq(sequence)
    seq_len = len(seq)

    for motif in motif_list:
        pwm = motif.counts.normalize(pseudocounts=pseudocount)
        pssm = pwm.log_odds()

        distribution = ScoreDistribution(pssm=pssm, background=_BACKGROUND)
        abs_threshold = distribution.threshold_fpr(fpr)

        for position, score in pssm.search(seq, threshold=abs_threshold):
            if position >= 0:
                strand = '+'
                fwd_position = position
            else:
                strand = '-'
                fwd_position = seq_len + position - len(motif)

            hits.append({
                'tf_name':      motif.name,
                'motif_id':     motif.matrix_id,
                'position':     fwd_position,
                'motif_length': len(motif),
                'strand':       strand,
                'score':        round(score, 4),
            })

    if not hits:
        return pd.DataFrame(columns=['tf_name', 'motif_id', 'position', 'motif_length', 'strand', 'score'])

    return (
        pd.DataFrame(hits)
        .sort_values('score', ascending=False)
        .reset_index(drop=True)
    )