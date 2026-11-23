import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X, y_true = make_blobs(
    n_samples=150,
    centers=[
        [25, 30],
        [60, 75],
        [80, 25]
    ],
    cluster_std=10,
    random_state=42
)

scatter = plt.scatter(
    X[:,0], X[:,1],
    s=70,
    c=y_true,
    cmap="viridis",
    alpha=0.8,
    edgecolors="black"
)

plt.title("Точечная диаграмма", fontsize=16)
plt.xlabel("X1")
plt.ylabel("X2")
plt.grid(True, alpha=0.3)

handles, labels = scatter.legend_elements()
plt.legend(handles, labels, title="Классы")

plt.tight_layout()
plt.show(block = False)

from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

kmeans.fit(X)

labels = kmeans.labels_

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=labels,
    cmap='viridis',
    s=50
)

centers = kmeans.cluster_centers_

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    c='red',
    marker='X',
    s=200,
    label='Центры кластеров'
)

plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Кластеризация покупателей')
plt.legend()

plt.show()

# Подсчет количества объектов в кластерах
unique, counts = np.unique(
    labels,
    return_counts=True
)

for cluster, count in zip(unique, counts):
    print(f'Кластер {cluster}: {count} покупателей')

inertia = []

for k in range(1, 10):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X)

    inertia.append(model.inertia_)

plt.plot(
    range(1, 10),
    inertia,
    marker='o'
)

plt.xlabel('Количество кластеров')
plt.ylabel('Inertia')
plt.title('Метод локтя')

plt.show()

