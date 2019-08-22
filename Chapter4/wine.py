import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.sbs import SBS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

warnings.simplefilter(action='ignore', category=FutureWarning)

df_wine = pd.read_csv('../data/wine.data', header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
                   'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color inensity',
                   'Hue', 'OD280/OD315 of diluted wines', 'Proline']

# Train Test split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# Normalize data
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

# Standardize data
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

lr = LogisticRegression(penalty='l1', C=1.0)
lr.fit(X_train_std, y_train)
# print('Training accuracy: ', lr.score(X_train_std, y_train))
# print('Test accuracy: ', lr.score(X_test_std, y_test))

fig = plt.figure(figsize=(13, 10))
ax = plt.subplot(111)
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'pink',
          'lightgreen', 'lightblue', 'gray', 'indigo', 'orange']
weights, params = [], []
for c in np.arange(-4., 6.):
    lr = LogisticRegression(penalty='l1', C=10.**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)
weights = np.array(weights)

# for column, color in zip(range(weights.shape[1]), colors):
#     plt.plot(params, weights[:, column], label=df_wine.columns[column + 1], color=color)
# plt.axhline(0, color='black', linestyle='--', linewidth=3)
# plt.xlim([10**(-5), 10**5])
# plt.ylabel('weight coefficient')
# plt.xlabel('C')
# plt.xscale('log')
# plt.legend(loc='upper left')
# ax.legend(loc='upper center', bbox_to_anchor=(1.15, 1.008), ncol=1, fancybox=True)
# plt.show()
#
# knn = KNeighborsClassifier(n_neighbors=5)
# sbs = SBS(knn, k_features=1)
# sbs.fit(X_train_std, y_train)
#
# k_feat = [len(k) for k in sbs.subsets_]
# plt.plot(k_feat, sbs.scores_, marker='o')
# plt.ylim([0.7, 1.02])
# plt.ylabel('Accuracy')
# plt.xlabel('Number of features')
# plt.grid()
# plt.show()

# k3 = list(sbs.subsets_[10])
# print(df_wine.columns[1:][k3])
#
# knn.fit(X_train_std, y_train)
# print('Training accuracy:', knn.score(X_train_std, y_train))
# print('Test accuracy:', knn.score(X_test_std, y_test))
# knn.fit(X_train_std[:, k3], y_train)
# print('Training accuracy:', knn.score(X_train_std[:, k3], y_train))
# print('Test accuracy:', knn.score(X_test_std[:, k3], y_test))

feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=500, random_state=1)
forest.fit(X_train, y_train)
'''
Importance values are normalized so that they sum up to 1.0
If two or more features are highly correlated, one feature may be ranked
very highly while the information of the other feature(s) may not be fully
captured. On the other hand, one don't need to be concerned about this problem
if one is merely interested in the predictive performance of a model rather
than the interpretation of feature importances values.
'''
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
# for f in range(X_train.shape[1]):
#     print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

sfm = SelectFromModel(forest, threshold=0.05, prefit=True)
X_selected = sfm.transform(X_train)
print('Number of features that this threshold criterion:', X_selected.shape[1])
for f in range(X_selected.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
