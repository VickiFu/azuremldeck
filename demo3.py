%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

iris = load_iris()
X, y = iris.data, iris.target
pca = PCA(n_components = 2) # keep 2 components which explain most variance
X = pca.fit_transform(X)

X.shape
#1

# I have to tell KMeans how many cluster centers I want
n_clusters  = 3

# for consistent results when running the methods below
random_state = 2
#2

'''
KMeans employ the Expectation-Maximization algorithm which works as follows: 
1.Guess cluster centers
2.Assign points to nearest cluster
3.Set cluster centers to the mean of points
4.Repeat 1-3 until converged
'''

def _kmeans_step(frame=0, n_clusters=n_clusters):
    rng = np.random.RandomState(random_state)
    labels = np.zeros(X.shape[0])
    centers = rng.randn(n_clusters, 2)

    nsteps = frame // 3

    for i in range(nsteps + 1):
        old_centers = centers
        if i < nsteps or frame % 3 > 0:
            dist = euclidean_distances(X, centers)
            labels = dist.argmin(1)

        if i < nsteps or frame % 3 > 1:
            centers = np.array([X[labels == j].mean(0)
                                for j in range(n_clusters)])
            nans = np.isnan(centers)
            centers[nans] = old_centers[nans]


    # plot the data and cluster centers
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='rainbow',
                vmin=0, vmax=n_clusters - 1);
    plt.scatter(old_centers[:, 0], old_centers[:, 1], marker='o',
                c=np.arange(n_clusters),
                s=200, cmap='rainbow')
    plt.scatter(old_centers[:, 0], old_centers[:, 1], marker='o',
                c='black', s=50)

    # plot new centers if third frame
    if frame % 3 == 2:
        for i in range(n_clusters):
            plt.annotate('', centers[i], old_centers[i], 
                         arrowprops=dict(arrowstyle='->', linewidth=1))
        plt.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c=np.arange(n_clusters),
                    s=200, cmap='rainbow')
        plt.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c='black', s=50)

    plt.xlim(-4, 5)
    plt.ylim(-2, 2)
    plt.ylabel('PC 2')
    plt.xlabel('PC 1')

    if frame % 3 == 1:
        plt.text(4.5, 1.7, "1. Reassign points to nearest centroid",
                 ha='right', va='top', size=8)
    elif frame % 3 == 2:
        plt.text(4.5, 1.7, "2. Update centroids to cluster means",
                 ha='right', va='top', size=8)
    else:
        plt.text(4.5, 1.7, "3. Updated",
                 ha='right', va='top', size=8)
#3

min_clusters, max_clusters = 1, 6
interact(_kmeans_step, frame=[0, 20],
                    n_clusters=[min_clusters, max_clusters])