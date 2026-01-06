import numpy as np
import pandas as pd
import scanpy as sc


def prepare_rna_seq(
    rna_seq_path: str,
    export_path: str,
    phenotype_path: str | None = None,
    num_top_varying_genes: int = 5000,
):
    rna_seq = sc.read_text(rna_seq_path, delimiter="\t").T
    rna_seq.X = np.round(np.exp2(rna_seq.X) - 1).astype(np.long)

    if phenotype_path is not None:
        phenotype = pd.read_csv(phenotype_path, delimiter="\t", index_col="sample")
        phenotype.rename(columns={"_primary_disease": "primary_disease"}, inplace=True)

        indices = rna_seq.obs_names.intersection(phenotype.index)
        rna_seq = rna_seq[indices, :]
        rna_seq.obs = phenotype.loc[indices]

    sc.pp.highly_variable_genes(
        rna_seq, n_top_genes=num_top_varying_genes, flavor="seurat_v3"
    )

    rna_seq.write_h5ad(export_path)
    return rna_seq
