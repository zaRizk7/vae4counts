# VAE for Counts

# TL;DR
This repo contains some of my exploration on modeling count data with VAEs, revisiting my [MSc project](https://github.com/zaRizk7/co-occurrence-model) on scene understanding. I found out that it's somewhat like an easier to model scRNA-seq data, where both data has similar data characteristics: high-dimensional and very sparse counts; albeit scRNA-seq is a more extreme case.

# Summary

Some of the things I found interesting about my MSc project was the emphasis on the interpretability and recognizing patterns from these types data in an unsupervised manner. There are various models I worked on, from mixture models, binary latent trees, to more niche models like Masked Autoencoder for Density Estimation (MADE) and Einsum Networks (EiNet). The first two models are nice as it is able to provide interpretability through its parameters or learned structure; the last two didn't provide much value apart from achieveing better inference performance. Variational Autoencoders (VAE) was one model suggested by my supervisor, but I decided to opt-in for the aforementioned models while my project partner opt for VAEs and discrete normalizing flows.

In this repo, I revisited some of the ideas from my MSc and combine it with concepts I found to be fascinating, interpretability and latent variable. Oddly enough, after several exploration, I discovered that my MSc project can be descibed as an easier version of modelling single-cell RNA-seq (scRNA-seq), which can have over tens of thousands of features, while my MSc project's data has only 80. Taking scRNA-seq modeling into account, I decided to implement VAEs inspired by scVI (Lopez et al., 2018) and scVAE (Grønbech et al., 2020) to model the count data I have. Given the nice outcome I have with mixture models, I decided to extend it to the Gaussian Mixture VAE (GMVAE) realm and use a linear decoder to have additional interpretation as described by (Grønbech et al., 2020) and (Svensson et al., 2020) respectively.

The outcome is having multiple layers of interpretability embedded into the model, including: latent space interpretation of VAEs, clusterable representation of mixture models, and linear factors of linear decoder; I found lots of interesting patterns compared to my original MSc project. I also tried applying it to both TCGA Pan-Cancer bulk RNA-seq (Aaltonen et al., 2018) and PBMC 68K scRNA-seq (Zheng et al., 2017), but needed to limit it to only 4096 highly variable genes/features given the limited computational resource of my device. The latter two seems to be more challenging in making a GM-distributed latent representation since I found out that the latent might be collapsing and focuses only on few components, even degenerates to a standard VAE.

If I have more time, I would try to extend the models to integrate multiple modalities. For my MSc project in scene recognition, I may integrate various modalities found in COCO dataset (Lin et al., 2014) including images and captions and leverage other omics from TCGA Pan-Cancer like DNA methylation and miRNA-seq.

# Setup

For prerequisite, I would recommend to install atleast Python 3.10+, since the repo has lots of static typing syntaxes that is supported from 3.10+. Note that I developed the code using Python 3.14.

There are several options to setup this repo. If you prefer to install it as a package you may use `pip` and do:
```sh
pip install "vae4counts@git+https://github.com/zaRizk7/vae4counts@main"
```
If you want to play around with the scene understanding notebook, you need to install optional dependencies and do it by:
```sh
pip install "vae4counts[coco-example]@git+https://github.com/zaRizk7/vae4counts@main"
```
For development, if wanted to be extended or modified, install it as a development package by doing:
```sh
git clone https://github.com/zaRizk7/vae4counts.git
cd vae4counts
uv pip install -e ".[dev]"
```

# Data Access

Both COCO 2017 object counts used for my MSc project and TCGA Pan-Cancer bulk RNA-seq can be downloaded and used directly through the functions implemented in `vae4counts.data` module. Be aware that TCGA Pan-Cancer data is quite large (~700MB+).

For PBMC 68K scRNA-seq, you need to access the data from 10X Genomics (https://www.10xgenomics.com/datasets/fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0) and download the Gene / cell matrix (filtered) data. The annotations seen in the visualization can be accessed at https://github.com/10XGenomics/single-cell-3prime-paper and download the `68k_pbmc_barcodes_annotation.tsv` file. Afterwards, you need to combine the matrix data and annotation together, which can be done through `scanpy` and `pandas`. Using `scanpy`, export the data into `.h5ad` and it should be usable for the notebook I produced. I'm not sure whether I can share the preprocessed `.h5ad` data without permissions, so I'm not going to share it for this repo.

# References

Lopez, R. et al. (2018) ‘Deep generative modeling for single-cell transcriptomics’, Nature Methods, 15(12), pp. 1053–1058. doi:10.1038/s41592-018-0229-2.

Grønbech, C.H. et al. (2020) ‘SCVAE: Variational auto-encoders for single-cell gene expression data’, Bioinformatics, 36(16), pp. 4415–4422. doi:10.1093/bioinformatics/btaa293.

Svensson, V. et al. (2020) ‘Interpretable factor models of single-cell RNA-seq via variational autoencoders’, Bioinformatics, 36(11), pp. 3418–3421. doi:10.1093/bioinformatics/btaa169.

Aaltonen, L.A., Abascal, F., Abeshouse, A., Aburatani, H., et al. (2020) Pan-cancer analysis of whole genomes. Nature. 578 (7793), 82–93. doi:10.1038/s41586-020-1969-6.

Zheng, G.X.Y., Terry, J.M., Belgrader, P., Ryvkin, P., Bent, Z.W., et al. (2017) Massively parallel digital transcriptional profiling of single cells. Nature Communications. 8 (1). doi:10.1038/ncomms14049.

Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P. & Zitnick, C.L. (2014) Microsoft COCO: Common Objects in Context. Lecture Notes in Computer Science.pp.740–755. doi:10.1007/978-3-319-10602-1_48.
