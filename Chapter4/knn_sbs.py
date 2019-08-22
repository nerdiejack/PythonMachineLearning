import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import utils.sbs

knn = KNeighborsClassifier(n_neighbors=5)

sbs = SBS(knn, k_)