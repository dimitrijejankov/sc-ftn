import pandas as pd
import numpy as np
import sklearn

from dataloader import load_test_data, load_train_data, features
from sklearn.cross_validation import train_test_split
from utils import rmspe
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

print("load features")

train = load_train_data()
test = load_test_data()

print('training data processing')

X_train, X_valid = train_test_split(train, test_size=0.012, random_state=10)
y_train = np.log1p(X_train.Sales)
y_valid = np.log1p(X_valid.Sales)

print("Train a model")

#X = train[features]
#Y = np.log1p(train.Sales)

clf = sklearn.ensemble.GradientBoostingRegressor(learning_rate=0.3,
                                                 subsample=1.0, max_depth=10, verbose=True)
#clf = sklearn.ensemble.ExtraTreesRegressor(n_estimators=100, max_depth=10, verbose=True)
#clf = SVR(kernel='rbf', C=1e3, gamma=0.1)
#clf = make_pipeline(PolynomialFeatures(4), Ridge())

clf.fit(X_train[features], y_train)

y_predict = clf.predict(X_valid[features])
print "Result is : " + str(rmspe(y_valid, y_predict))

y_predict = clf.predict(test[features])
indices = y_predict < 0
y_predict[indices] = 0
submission = pd.DataFrame({"Id": test["Id"], "Sales": np.exp(y_predict) - 1})
submission.to_csv("sk_submission.csv", index=False)


