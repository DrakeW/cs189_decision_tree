from decision_tree import DecisionTree
from random_forest import RandomForest
import numpy as np
from scipy import io

data = io.loadmat('spam_dist/spam_data.mat')

train_data = data['training_data']
train_lab = data['training_labels'][0]

test_data = data['test_data']

np.random.seed(123)
valid_idx = np.random.choice(len(train_data), int(0.2 * len(train_data)), replace=False)

valid_data = train_data[valid_idx]
valid_lab = train_lab[valid_idx]

train_data = np.delete(train_data, valid_idx, 0)
train_lab = np.delete(train_lab, valid_idx)

# # Decision Tree
clf = DecisionTree(max_depth=20)
clf.train(train_data, train_lab)

preds = clf.predict(valid_data)
print np.mean(preds == valid_lab) # 0.797890295359

print clf.depth

# Random Forest
# rf_clf = RandomForest(100, train_data.shape[0] / 4, int(np.sqrt(train_data.shape[1])), max_depth=20)
# rf_clf.train(train_data, train_lab)
#
# preds2 = rf_clf.predict(valid_data)
# print np.mean(preds2 == valid_lab)
