import lazypredict
from lazypredict.Supervised import LazyClassifier #CLassification
from lazypredict.Supervised import LazyRegressor #Regression

from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split # data split


# load data
data = datasets.load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)


# fit all models
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)

print(models)

#Print complete list of classifiers
lazypredict.Supervised.CLASSIFIERS

#Selectig initial 10 classifiers
lazypredict.Supervised.CLASSIFIERS = lazypredict.Supervised.CLASSIFIERS[:10]
lazypredict.Supervised.CLASSIFIERS


# fit selected models
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)
models

# load data
boston = datasets.load_iris()
print(boston.DESCR)
print(boston.feature_names)
print(boston.target_names)
print(boston.data.shape)
print(boston.target.shape)
print(boston.data[::5])
print(boston.target[::5])
# print(boston.count_nonzero())
# print(boston.head())
print(boston)

X, y = boston.data, boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)


# fit all models
reg = LazyRegressor(predictions=True)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

print(models)

#Print complete list of regressor
lazypredict.Supervised.REGRESSORS


# fit selected models
clf = LazyRegressor(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)
models
