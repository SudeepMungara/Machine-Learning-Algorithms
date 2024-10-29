from utils import euclidean_distance
import numpy as np
from collections import Counter
class KNN:

    def __init__(self,k):
        self.K = k

    def fit(self,X,y):
        self.X_train = X
        self.y_train = y
    
    def predict(self,X):
        preds = [self._predict_distance(x) for x in X]
        return preds

    def _predict_distance(self,x):
        k = min(self.K, len(self.X_train))
        distance =[euclidean_distance(x,x_train) for x_train in self.X_train]
        k_idxs = np.argsort(distance)[:k]
        k_nearest_labels = [self.y_train[i] for i in k_idxs]
        return Counter(k_nearest_labels).most_common()[0][0]
