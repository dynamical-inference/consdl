import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import DBSCAN
import umap


def umap_dbscan(embeddings, umap_n=4, umap_seed=42, dbscan_eps=0.3, dbscan_min_samples=10):
    """
    Reduce embeddings to umap_n and cluster using DBSCAN.
    Visualized 2D
    """
    print(f"UMAP: Reducing {embeddings.shape} into ({embeddings.shape[0]}, {umap_n}) ...")
    umap_reducer = umap.UMAP(n_components=umap_n, random_state=umap_seed)
    embeddings_reduced = umap_reducer.fit_transform(embeddings)

    print("DBSCAN: Clustering reduced embedings ...")
    dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    clusters = dbscan.fit_predict(embeddings_reduced)

    print("Visualize in 2D")
    if umap_n == 2:
        embeddings_2d = embeddings_reduced
    else:
        umap_reducer = umap.UMAP(n_components=2, random_state=umap_seed)
        embeddings_2d = umap_reducer.fit_transform(embeddings)

    df = pd.DataFrame({"x": embeddings_2d[:, 0], "y": embeddings_2d[:, 1], "cluster": clusters})
    df["index"] = np.arange(len(df))

    fig = px.scatter(df, x="x", y="y", color="cluster", hover_data=["index"], title="UMAP 2D")
    fig.show()

    return embeddings_reduced, clusters
