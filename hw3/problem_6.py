from sklearn.decomposition import PCA


def fitPCA(data):
    pca = PCA(n_components=3)
    x = pca.fit_transform(data)
    return x