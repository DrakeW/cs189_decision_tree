import numpy as np
import scipy.stats as stats

class DTNode:
    def __init__(self, feature, threshold, left, right, label=None):
        self.split_rule = (feature, threshold)
        self.left = left
        self.right = right
        self.label = label


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None

    def train(self, train_data, train_labels):
        self.root = self.build_tree(train_data, train_labels)

    def predict(self, data):
        return map(lambda p: self.pred_data_point(p), data)

    def pred_data_point(self, data_point):
        node = self.root
        while node.label is None:
            feat, val = node.split_rule
            if data_point[feat] < val:
                node = node.left
            else:
                node = node.right
        return node.label

    """
    build tree build the decision tree and return the root node
    """
    # TODO: need to handle max depth here
    def build_tree(self, data, labels):
        if len(labels) == 0:
            return None
        if len(set(labels)) == 1:
            return DTNode(None, None, None, None, labels[0])
        split_rule = self.segmenter(data, labels)
        if split_rule == None:
            return DTNode(None, None, None, None, np.argmax(np.bincount(labels)))
        feature, val = split_rule
        left_idx, right_idx = data[:,feature] < val, data[:, feature] >= val
        return DTNode(feature, val, \
                      self.build_tree(data[left_idx], labels[left_idx]), \
                      self.build_tree(data[right_idx], labels[right_idx]))

    """
    return best feature to split
    """
    def segmenter(self, data, labels):
        best_split_rule, smallest_impurity = None, float("inf")
        for feat in range(data.shape[1]):
            # TODO: use radix sort if that can make it faster
            values = np.sort(list(set(data[:,feat])))
            for idx, val in enumerate(values):
                if idx == len(values) - 1:
                    continue
                split_val = float(values[idx] + values[idx+1])/2
                left = labels[data[:,feat] < split_val]
                right = labels[data[:, feat] >= split_val]
                imp = self.impurity(self.get_hist(left), self.get_hist(right))
                if imp < smallest_impurity:
                    smallest_impurity = imp
                    best_split_rule = (feat, split_val)
        return best_split_rule

    def get_hist(self, labels):
        return np.unique(labels, return_counts=True)[1] / float(len(labels))

    """
    return the badness (to minimize) of a split
    """
    def impurity(self, left_label_hist, right_label_hist):
        H_l, H_r = self.entropy(left_label_hist), self.entropy(right_label_hist)
        len_l, len_r = len(left_label_hist), len(right_label_hist)
        return (len_l * H_l + len_r * H_r) / (len_l + len_r)

    def entropy(self, freq_data):
        return -1 * np.sum(freq_data * np.log2(freq_data))
