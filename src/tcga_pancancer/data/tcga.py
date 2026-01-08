from pathlib import Path
from urllib.request import urlretrieve

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

__all__ = ["load_tcga_pancancer_rna_seq"]

BASE_URL = "https://toil-xena-hub.s3.us-east-1.amazonaws.com/download"

URL = {
    "rna_seq": (
        f"https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/tcga_gene_expected_count.gz",
        "rna-seq.tsv.gz",
    ),
    "phenotype": (
        f"https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/TCGA_phenotype_denseDataOnlyDownload.tsv.gz",
        "phenotype.tsv.gz",
    ),
}


def _load_preprocessed_tcga_rna_seq(data_path: Path, num_top_genes: int | None = None):
    adata = ad.read_h5ad(data_path)
    num_top_genes = num_top_genes if num_top_genes is not None else adata.n_vars
    sc.pp.highly_variable_genes(adata, n_top_genes=num_top_genes, flavor="seurat_v3")
    adata = adata[:, adata.var["highly_variable"]]
    return adata


def _download_tcga_raw_data(data_path: Path):
    raw_path = data_path / "raw"
    raw_path.mkdir(exist_ok=True)

    for k in URL:
        url, filename = URL[k]
        fpath = raw_path / filename
        if not fpath.exists():
            urlretrieve(url, fpath)


def _preprocess_raw_tcga_pancancer(data_path: Path):
    rna_seq = pd.read_csv(
        data_path / "raw" / URL["rna_seq"][1],
        index_col="sample",
        sep="\t",
        compression="gzip",
    )
    rna_seq = rna_seq.transpose()

    # Apply exp2m1 on the log2 count data
    rna_seq = rna_seq.apply(
        lambda count: np.round(np.expm1(count * np.log(2))).astype(np.long)
    )

    # Preprocess phenotypes
    phenotypes = pd.read_csv(
        data_path / "raw" / URL["phenotype"][1], sep="\t", index_col="sample"
    )

    phenotypes.rename(columns={"_primary_disease": "primary_disease"}, inplace=True)
    phenotypes["sample_type_id"] = (
        phenotypes["sample_type_id"].fillna(-1).astype(np.long)
    )

    # Align indices
    index = rna_seq.index.intersection(phenotypes.index)
    rna_seq, phenotypes = rna_seq.loc[index], phenotypes.loc[index]

    # Make folder
    (data_path / "preproc").mkdir(exist_ok=True)

    # Return as anndata
    adata = sc.AnnData(rna_seq, phenotypes)
    adata.write_h5ad(data_path / "preproc" / "rna-seq.h5ad")
    return adata


def load_tcga_pancancer_rna_seq(
    data_path: str = "tcga-pancancer", num_top_genes: int | None = None
):
    data_path: Path = Path(data_path)
    if not data_path.exists():
        data_path.mkdir()

    preproc_path = data_path
    if preproc_path.is_dir():
        preproc_path = preproc_path / "preproc" / "rna-seq.h5ad"

    if (
        preproc_path.exists()
        and preproc_path.is_file()
        and preproc_path.suffix == ".h5ad"
    ):
        return _load_preprocessed_tcga_rna_seq(preproc_path, num_top_genes)

    # Download missing data
    _download_tcga_raw_data(data_path)
    _preprocess_raw_tcga_pancancer(data_path)

    # Preprocess TCGA
    return _load_preprocessed_tcga_rna_seq(preproc_path, num_top_genes)
