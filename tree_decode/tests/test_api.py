from sklearn.tree import DecisionTreeClassifier
from sklearn.exceptions import NotFittedError

import tree_decode.api as api
import numpy as np
import pytest


class TestGetTreeInfo(object):

    def test_unsupported(self):
        match = "Function support is only implemented for"
        message = "Expected NotImplementedError regarding no support"

        with pytest.raises(NotImplementedError, match=match, message=message):
            api.get_tree_info([])

    def test_unfitted(self):
        match = "instance is not fitted yet"
        message = "Expected NotFittedError regarding fitting"

        with pytest.raises(NotFittedError, match=match, message=message):
            api.get_tree_info(DecisionTreeClassifier())


class TestGetDecisionInfo(object):

    def test_unsupported(self):
        match = "Function support is only implemented for"
        message = "Expected NotImplementedError regarding no support"

        with pytest.raises(NotImplementedError, match=match, message=message):
            api.get_decision_info([], np.array([]))

    def test_unfitted(self):
        match = "instance is not fitted yet"
        message = "Expected NotFittedError regarding fitting"

        with pytest.raises(NotFittedError, match=match, message=message):
            api.get_decision_info(DecisionTreeClassifier(), np.array([]))
