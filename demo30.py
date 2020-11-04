import pandas as pd
from sklearn import datasets
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns = iris.feature_names)
y = iris.target

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(df, y)
dtree

import graphviz as gv
from sklearn import tree
dot_data = tree.export_graphviz(dtree, out_file=None, feature_names = iris.feature_names, filled=True,
                               rounded=True, special_characters=True)
graph = gv.Source(dot_data)
#graph