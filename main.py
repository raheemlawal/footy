import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score
import numpy as np

#load csv into pandas DataFrame object
df = pd.read_csv('all_players.csv')

#preprocessing, players must have played atleast 10 games
df = df[(df['GP'] > 20)] 

'''
#number of rows and columns
r, c = df.shape
print(f'({r}, {c})') 
'''

#splitting data

#x = df.drop(['POS','Year', 'Club'], axis=1)
#x = x.select_dtypes(include=np.number)

x = df[['OFF', 'G', 'YC', 'GWG', 'SHTS']]
y = df['POS']

#action: creating training and test sets (75/25)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=100)


#run KNN model
#sqrt(rows) = ~122)

knn = KNeighborsClassifier(n_neighbors=122)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)