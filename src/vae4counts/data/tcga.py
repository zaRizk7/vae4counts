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
        "https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/tcga_gene_expected_count.gz",
        "rna-seq.tsv.gz",
    ),
    "phenotype": (
        "https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/TCGA_phenotype_denseDataOnlyDownload.tsv.gz",
        "phenotype.tsv.gz",
    ),
}


def _load_preprocessed_tcga_rna_seq(data_path: Path, num_top_genes: int | None = None):
    """Loads the preprocessed `.h5ad` TCGA Pan-Cancer RNA-seq data.

    Args:
        data_path (str | Path): Directory to the preprocessed dataset.
        num_top_genes (str): Number of top genes ranked by largest variance. If None,
            selects all of the genes. Default: None

    Returns:
        ad.AnnData: Annotated data matrix with the count data in adata.X and phenotypes
            stored in `adata.obs`.
    """
    adata = ad.read_h5ad(data_path)
    num_top_genes = num_top_genes if num_top_genes is not None else adata.n_vars
    sc.pp.highly_variable_genes(adata, n_top_genes=num_top_genes, flavor="seurat_v3")
    adata = adata[:, adata.var["highly_variable"]]
    return adata


def _download_tcga_raw_data(data_path: Path):
    """Downloads TCGA Pan-Cancer raw data. The two downloaded data are
    the raw log2(1+count) normalized data and phenotype data. Both are in
    `tsv.gz` format. The downloaded data will be placed in `data_path / "raw"`.

    Args:
        data_path (str): Directory to download the raw data.
    """
    raw_path = data_path / "raw"
    raw_path.mkdir(exist_ok=True)

    for k in URL:
        url, filename = URL[k]
        fpath = raw_path / filename
        if not fpath.exists():
            urlretrieve(url, fpath)


def _preprocess_raw_tcga_pancancer(data_path: Path):
    """Preprocess TCGA Pan-Cancer raw data. Steps includes:
        1. Transpose from (gene x cell) -> (cell x gene).
        2. Invert from `log2(1+count)` -> `exp2(count - 1)`.
        3. Parses phenotype data and match cell ids between RNA-seq and phenotype.
        4. Stores as a `.h5ad` file to `data_path / "preproc" / "rna-seq.h5ad"`

    Args:
        data_path (str): Directory to the dataset folder containing "raw" folder and
            raw format datasets.
    """
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
    """Loads the TCGA Pan-Cancer RNA-seq with its phenotypes. If the data is not found,
    it will be downloaded automatically. There are a total of 10.5K cells (or samples) with
    over 60K of recorded genes (or feature). Hence, it is recommended to take the highly
    varying genes given the large dimension of the data.

    Args:
        data_path (str | Path): Directory to the dataset. Default: "coco2017-object-counts"
        num_top_genes (str): Number of top genes ranked by largest variance. If None,
            selects all of the genes. Default: None

    Returns:
        ad.AnnData: Annotated data matrix with the count data in adata.X and various metadata
            for observations (in `adata.obs`) and features (in `adata.var`).
    """
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
