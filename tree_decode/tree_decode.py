import numpy as np

from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import normalize


def get_tree_info(estimator):
    check_is_fitted(estimator, "tree_")
    tree = estimator.tree_

    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right

    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold

    node_depths = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)

    stack = [(0, -1)]

    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depths[node_id] = parent_depth + 1

        if children_left[node_id] != children_right[node_id]:
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    print_tab = " " * 5

    previous_leaf = False
    previous_depth = -1

    for i in range(n_nodes):
        node_depth = node_depths[i]
        tabbing = node_depth * print_tab

        if is_leaves[i]:
            if previous_leaf:
                if previous_depth > 0 and previous_depth > node_depth:
                    print("")  # Readability

            probs = tree.value[i][0]
            probs = normalize(np.atleast_2d(probs),
                              norm="l1")
            probs = probs.round(3)

            leaf_info = "{tabbing}node={label} left node: scores = {score}"
            print(leaf_info.format(tabbing=tabbing, label=i, score=probs))

            previous_depth = node_depth
            previous_leaf = True
        else:
            if previous_leaf:
                previous_leaf = False
                print("")  # Readability

            node_info = ("{tabbing}node={label}: go to node {left} if "
                         "feature {name} <= {cutoff} else to node {right}.")
            print(node_info.format(tabbing=tabbing, label=i,
                                   left=children_left[i],
                                   name=feature[i],
                                   cutoff="%.2f" % threshold[i],
                                   right=children_right[i]))


def demo():
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.datasets import load_iris

    iris = load_iris()
    y = iris.target
    x = iris.data

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
    estimator = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)

    estimator.fit(x_train, y_train)
    get_tree_info(estimator)


if __name__ == "__main__":
    demo()
