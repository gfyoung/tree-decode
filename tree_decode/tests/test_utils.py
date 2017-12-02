from sklearn.tree import DecisionTreeClassifier

import tree_decode.utils as utils
import pytest


def test_check_estimator_type():
    estimator = DecisionTreeClassifier()
    utils.check_estimator_type(estimator)

    match = "Function support is only implemented for"
    message = "Expected NotImplementedError regarding no support"

    with pytest.raises(NotImplementedError, match=match, message=message):
        utils.check_estimator_type([])
