from sklearn.cluster import KMeans as sk_KMeans
import numpy as np


def test_kmeans():
    X = np.array([[1, 2], [1, 4], [1, 0],
                 [10, 2], [10, 4], [10, 0]])
    sk_model = sk_KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)

    pred = sk_model.predict([[0, 0], [12, 3]])
    print(pred)
    assert True
    #sk_model.cluster_centers_

