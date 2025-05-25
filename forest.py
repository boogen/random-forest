import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data():
  df = pd.read_csv('train.csv')
  df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

  df['Age'] = df['Age'].fillna(df['Age'].median())
  df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

  df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
  df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

  X = df.drop('Survived', axis=1).values
  y = df['Survived'].values

  return train_test_split(X, y, test_size=0.5, random_state=42)

def gini(y):
  classes = np.unique(y)
  return 1.0 - sum((np.sum(y == c) / len(y)) ** 2 for c in classes)

def gini_gain(y, y_left, y_right):
  p = len(y_left) / len(y)
  return gini(y) - p * gini(y_left) - (1 - p) * gini(y_right)

def split(X, y, feature, threshold):
  left = X[:, feature] <= threshold
  return X[left], y[left], X[~left], y[~left]

def best_split(X, y):
  best_feat, best_thresh, best_gain = None, None, 0
  features = X.shape[1]
  for feature in range(features):
    thresholds = np.unique(X[:, feature])
    for t in thresholds:
      X_left, y_left, X_right, y_right = split(X, y, feature, t)
      if len(y_left) == 0 or len(y_right) == 0:
        continue
      gain = gini_gain(y, y_left, y_right)
      if gain > best_gain:
        best_feat, best_thresh, best_gain = feature, t, gain
  return best_feat, best_thresh

class DecisionTree:
  def __init__(self, max_depth=10, min_samples_split=2):
    self.max_depth = max_depth
    self.min_samples_split = min_samples_split

  def fit(self, X, y, depth=0):
    if depth >= self.max_depth or len(set(y)) == 1 or len(y) < self.min_samples_split:
      self.prediction = np.bincount(y).argmax()
      return self

    f, t = best_split(X, y)
    if f is None:
      self.prediction = np.bincount(y).argmax()
      return self

    self.feature = f
    self.threshold = t

    left_mask = X[:, f] <= t

    self.left_tree = DecisionTree(self.max_depth, self.min_samples_split).fit(X[left_mask], y[left_mask], depth + 1)
    self.right_tree = DecisionTree(self.max_depth, self.min_samples_split).fit(X[~left_mask], y[~left_mask], depth + 1)

    return self

  def predict_single(self, x):
    if hasattr(self, 'prediction'):
      return self.prediction
  
    branch = self.left_tree if x[self.feature] <= self.threshold else self.right_tree
    return branch.predict_single(x)

  def predict(self, X):
    return np.array([self.predict_single(x) for x in X])



class RandomForest:
  def __init__(self, n_trees=10, max_depth=10, min_samples_split=2):
    self.n_trees = n_trees
    self.max_depth = max_depth
    self.min_samples_split = min_samples_split
    self.trees = []

  def fit(self, X, y):
    n_samples = X.shape[0]
    for _ in range(self.n_trees):
      indices = np.random.choice(n_samples, n_samples, replace=True)
      X_sample, y_sample = X[indices], y[indices]
      tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
      tree.fit(X_sample, y_sample)
      self.trees.append(tree)
  
  def predict(self, X):
    tree_preds = np.array([tree.predict(X) for tree in self.trees])
    return np.round(np.mean(tree_preds, axis=0)).astype(int)

X_train, X_test, y_train, y_test = load_data()
forest = RandomForest(n_trees=20, max_depth=10)
forest.fit(X_train, y_train)
preds = forest.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, preds))



