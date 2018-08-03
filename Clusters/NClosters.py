import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.DataFrame()
df = pd.read_csv("DataE.csv", encoding = "ISO-8859-1")

df = df[['MeGusta','MeEncanta','MeDivierte','MeAsombra','MeEntristece','MeEnoja','PAN','PRI','MORENA','INDEPENDIENTE','POSITIVO','NEGATIVO','NEUTRAL']]
distortions = []
for i in range(1, 11):
     km = KMeans(n_clusters=i,
                 init='k-means++',
                 n_init=10,
                 max_iter=300,
                 random_state=0)
     km.fit(df)
     distortions.append(km.inertia_)
plt.plot(range(1,11), distortions, marker='*',c="red"  )
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()