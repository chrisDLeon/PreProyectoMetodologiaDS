import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd

clusters = KMeans(n_clusters=4, #Numero especificado de clusters
                  init='k-means++',
                  n_init=10,    #Corre el programa con diferentes centros random
                  max_iter=300, #maximo numero de iteraciones
                  tol=1e-04,    #parametro que controla la tolerancia
                  random_state=0)
df = pd.DataFrame()
df = pd.read_csv("DataE.csv", encoding = "ISO-8859-1")
df = df[['MeGusta','MeEncanta','MeDivierte','MeAsombra','MeEntristece','MeEnoja','PAN','PRI','MORENA','INDEPENDIENTE','POSITIVO','NEGATIVO','NEUTRAL']]

df = normalize(df,norm='l2',axis=1,copy=True,return_norm=False) #Se normaliza para eliminar datos redundantes e irregulares

clusters.fit(df)

new_point = np.array([[0,15,0,100,1,100,0,0,1,0,1,0,0]])
print(clusters.predict(new_point))




