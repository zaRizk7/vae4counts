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
    data_path = Path(data_path)
    url, filename = URL[subset]
    data_path.mkdir(exist_ok=True)

    csv_path = data_path / filename
    if not csv_path.exists():
        urlretrieve(url, data_path / filename)

    df = pd.read_csv(csv_path, index_col="id")
    metadata = df[METADATA]
    object_counts = df.drop(columns=METADATA)

    return ad.AnnData(object_counts, metadata)
