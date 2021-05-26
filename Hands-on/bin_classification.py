import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.model_selection import cross_val_score, cross_val_predict

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
print(mnist.keys())

X, y = mnist["data"], mnist["target"]

""" showing digit """
# some_digit = X[0]
# some_digit_image = some_digit.reshape(28, 28)
# plt.imshow(some_digit_image, cmap=mpl.cm.binary, interpolation="nearest")
# plt.axis("off")
# plt.show()

y = y.astype(np.uint8)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
# for element, tar in zip(X[:10], y[:10]):
#     print(sgd_clf.predict([element]))
#     print(tar)

print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))

""" confusion matrix """
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
print(confusion_matrix(y_train_5, y_train_pred))
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]  # ~7816
y_train_pred_90 = (y_scores >= threshold_90_precision)

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method="predict_proba")
y_scores_forest = y_probas_forest[:, 1]
# score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)