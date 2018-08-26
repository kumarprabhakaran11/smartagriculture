import numpy as np
from sklearn.dataset
from sklearn import tree
import pydotplus
import pandas as pd

# collect training data
df = pd.read_csv("C:\Users\admin\Desktop\train.csv")

iris = load_iris()
test_idx = [0, 738, 9]

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# train classifier

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

# make predictions

print(test_target) #expected outcome
print(clf.predict(test_data)) #model predicted outcome

# visualise tree

# non-coloured version
dot_data = tree.export_graphviz(clf, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("iris.pdf")


