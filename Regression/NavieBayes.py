import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


df = pd.DataFrame()
df = pd.read_csv("DataE.csv",encoding = "ISO-8859-1")


train,test = train_test_split(df, test_size = 0.2,random_state=1)

train_X = train[['PRI','PAN','MORENA','INDEPENDIENTE']]
train_y=train.POSITIVO
test_X= test[['PRI','PAN','MORENA','INDEPENDIENTE']]
test_y=test.POSITIVO

nb = GaussianNB()
nb.fit(train_X,train_y)
p = nb.predict(test_X)

new_point = np.array([[0,1,1,1]])
print(nb.predict(new_point))

print('Accuracy :',accuracy_score(p,test_y))
print('Precision :',precision_score(p,test_y))
print('Recall :', recall_score(p,test_y))
