import numpy as np
from decision_tree import DecisionTree

class RandomForest:
    def __init__(self, num_trees, sample_size, feature_size, max_depth=None):
        self.decision_trees = []
        self.num_trees = num_trees
        self.sample_size = sample_size
        self.feature_size = feature_size
        self.max_depth = max_depth

    def train(self, train_data, train_labels):
        for i in range(self.num_trees):
            # data bagging
            sample_idx = np.random.randint(0, train_data.shape[0], self.sample_size)
            t_data, t_labels = train_data[sample_idx], train_labels[sample_idx]
            # attr bagging
            dtree = DecisionTree(max_depth=self.max_depth)
            dtree.train(t_data, t_labels, attr_bagging_size=self.feature_size)
            self.decision_trees.append(dtree)

    def predict(self, data):
        return map(lambda p: self.pred_data_point(p), data)

    def pred_data_point(self, data_point):
        tree_preds = map(lambda dt: dt.pred_data_point(data_point), self.decision_trees)
        return np.argmax(np.bincount(tree_preds))
