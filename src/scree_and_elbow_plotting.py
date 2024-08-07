import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def pca_scree_plot(X, title = "", max_dim = 10, min_dim = 1, figsize = (8,6)):
    pca = PCA(n_components = max_dim, random_state = 0)
    pca.fit(X)
    fontsize = 20
    mpl.rcParams['font.size'] = fontsize
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(111)
    ax.plot(np.arange(1,max_dim+1), pca.explained_variance_ratio_, marker = 'o', markersize = 15, linestyle = '--',linewidth = 5, color = "blue")
    ax.set_xlabel('Component Number')
    ax.set_ylabel('Proportion of Variance Explained')
    ax.set_xticks(np.arange(1,max_dim+1))
    ax.set_title(title)
    fig.tight_layout()
    plt.show()
    return fig, ax

def kmeans_elbow_plot(X_train, title = "", max_clusters = 10, min_clusters = 2, figsize = (8,6)):
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