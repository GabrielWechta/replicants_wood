import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
print(mnist.keys())

X, y = mnist["data"], mnist["target"]
some_digit = X[0]
some_digit.reshape(28, 28)
y = y.astype(np.uint8)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# sgd_clf = SGDClassifier(random_state=42)
# sgd_clf.fit(X_train, y_train)
# sgd_clf.predict([some_digit])

# forest_clf = RandomForestClassifier(random_state=42)
# forest_clf.fit(X_train, y_train)
# print(forest_clf.predict([some_digit]))
# print(forest_clf.predict_proba([some_digit]))
#
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
# y_train_pred = cross_val_predict(forest_clf, X_train_scaled, y_train, cv=3)
# conf_mx = confusion_matrix(y_train, y_train_pred)
# plt.matshow(conf_mx, cmap=plt.cm.gray)
# plt.show()
#
# row_sums = conf_mx.sum(axis=1, keepdims=True)
# norm_conf_mx = conf_mx / row_sums
# np.fill_diagonal(norm_conf_mx, 0)
# plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
# plt.show()

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
print(knn_clf.predict([some_digit]))