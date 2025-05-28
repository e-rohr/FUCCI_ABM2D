import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def pca_scree_plot(X, title="", max_dim=10, min_dim=1, figsize=(8, 6)):
    """
    Generates a scree plot showing the proportion of variance explained by each principal component.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The input data matrix for PCA.
    title : str, optional
        Title of the plot (default is "").
    max_dim : int, optional
        Maximum number of principal components to display (default is 10).
    min_dim : int, optional
        Minimum number of principal components (not currently used but can be used to truncate the plot).
    figsize : tuple, optional
        Figure size for the plot (default is (8, 6)).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the scree plot.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object of the scree plot.

    Notes
    -----
    - The scree plot displays the explained variance ratio for each component.
    - PCA is fit using `sklearn.decomposition.PCA` with a fixed random seed for reproducibility.
    - `min_dim` is accepted but not used in the plot range.
    """

    pca = PCA(n_components = max_dim, random_state = 0)
    pca.fit(X)
    fontsize = 20
    mpl.rcParams['font.size'] = fontsize
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(111)
    ax.plot(np.arange(1,max_dim+1), pca.explained_variance_ratio_, marker = 'o', markersize = 15, linestyle = '--',linewidth = 5, color = "blue")
    ax.set_xlabel('Number of Principal Components')
    ax.set_ylabel('Proportion of Variance Explained')
    ax.set_xticks(np.arange(1,max_dim+1))
    ax.set_title(title)
    fig.tight_layout()
    plt.show()
    return fig, ax

def kmeans_elbow_plot(X_train, title="", max_clusters=10, min_clusters=2, figsize=(8, 6)):
    """
    Generates an elbow plot to help determine the optimal number of clusters for KMeans.

    Parameters
    ----------
    X_train : ndarray of shape (n_samples, n_features)
        The input data for KMeans clustering.
    title : str, optional
        Title of the plot (default is "").
    max_clusters : int, optional
        Maximum number of clusters to evaluate (default is 10).
    min_clusters : int, optional
        Minimum number of clusters to evaluate (default is 2).
    figsize : tuple, optional
        Figure size for the plot (default is (8, 6)).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the elbow plot.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object of the elbow plot.

    Notes
    -----
    - Uses the KMeans `inertia_` as a measure of within-cluster sum of squares.
    - KMeans is run with `n_init='auto'` and `random_state=0` for reproducibility.
    """

    score_vec = [] 
    for K in np.arange(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters = K,n_init = 'auto', random_state = 0)
        kmeans.fit(X_train)
        score_vec.append(kmeans.inertia_)
    
    fontsize = 20
    mpl.rcParams['font.size'] = fontsize   
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(111)
    ax.plot(np.arange(min_clusters, max_clusters + 1), score_vec, marker = 'o', markersize = 15, linestyle = '--',linewidth = 5, color = 'orange')
    ax.set_xlabel('Number of Clusters', fontsize = fontsize)
    ax.set_xticks(np.arange(min_clusters, max_clusters + 1))
    ax.set_ylabel('Sum of Squared Errors')
    ax.set_title(title)
    fig.tight_layout()
    plt.show()
    return fig, ax