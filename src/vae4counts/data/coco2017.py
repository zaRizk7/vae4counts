from pathlib import Path
from urllib.request import urlretrieve

import anndata as ad
import pandas as pd

__all__ = ["load_coco2017_object_counts"]

BASE_URL = "https://github.com/zaRizk7/co-occurrence-model/raw/refs/heads/main/co-occurrence-model/dataset"

URL = {
    "train": (f"{BASE_URL}/coco2017-cooccurences-train.csv", "train.csv"),
    "valid": (f"{BASE_URL}/coco2017-cooccurences-valid.csv", "valid.csv"),
}


METADATA = [
    "license",
    "file_name",
    "coco_url",
    "height",
    "width",
    "date_captured",
    "flickr_url",
]


def load_coco2017_object_counts(
    data_path: str | Path = "coco2017-object-counts", subset: str = "train"
):
    """Loads count data extracted from COCO 2017 images. If the data is not found,
    it will be downloaded automatically. The training and validation set has over
    120K and 5K samples respectively.

    Args:
        data_path (str | Path): Directory to the dataset. Default: "coco2017-object-counts"
        subset (str): Dataset subset, either "train" or "valid". Default: "train"

    Returns:
        ad.AnnData: Annotated data matrix with the count data in adata.X and various metadata
            for observations (in `adata.obs`) and features (in `adata.var`).
    """
    data_path = Path(data_path)
    url, filename = URL[subset]
    data_path.mkdir(exist_ok=True)

    # Download data if not exists
    csv_path = data_path / filename
    if not csv_path.exists():
        urlretrieve(url, data_path / filename)

    # Load data and separate count and metadata
    df = pd.read_csv(csv_path, index_col="id")
    metadata = df[METADATA]
    object_counts = df.drop(columns=METADATA)

    # Separate object name and category from raw column naming [object name]-[category]
    object_names, categories = [], []
    for col in object_counts.columns:
        object_name, category = col.split("]-[")
        object_names.append(object_name.replace("[", "").replace(" ", "_"))
        categories.append(category.replace("]", ""))

    # Produce var description for anndata
    object_counts.columns = object_names
    var = pd.DataFrame({"category": categories}, object_names)

    return ad.AnnData(object_counts, metadata, var)
