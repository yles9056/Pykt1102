# make a directory data
# copy sonar-all to data
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

df = pd.read_csv('data/sonar.all-data', header=None, prefix='X')
print('df.shape', df.shape)
data, labels = df.iloc[:, :-1], df.iloc[:, -1]
print('data.shape', data.shape)
print('labels.shape', labels.shape)
df.rename(columns={'X60': "Label"}, inplace=True)
print('df.columns\n', df.columns)
clf = KNeighborsClassifier(n_neighbors=6)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
print("score", clf.score(X_test, y_test))
result_cm1 = confusion_matrix(y_test, y_predict)
print('result_cm1\n', result_cm1)

scores = cross_val_score(clf, data, labels, cv=3, groups=labels)
print('cross_val_score', scores)

from joblib import dump, load

# 儲存fit好的模型
dump(clf, "knn1.joblib")
knn2 = load('knn1.joblib')
y_predict2 = knn2.predict(X_test)  # 讀取模型後重新預測
result2 = confusion_matrix(y_predict, y_predict2)  # 驗證讀取模型的預測結果是否和原來的模型預測結果一樣
print('confusion_matrix\n', result2)
