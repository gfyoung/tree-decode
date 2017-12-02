from sklearn.tree import DecisionTreeClassifier
from sklearn.exceptions import NotFittedError

import tree_decode.api as api
import numpy as np
import pytest


class BaseApiTest(object):

    min_args = ()

    @staticmethod
    def api_call(*args, **kwargs):
        raise NotImplementedError("API calling not implemented for base class")

    def test_unsupported(self):
        match = "Function support is only implemented for"
        message = "Expected NotImplementedError regarding no support"

        with pytest.raises(NotImplementedError, match=match, message=message):
            self.api_call([], *self.min_args)

    def test_unfitted(self):
        match = "instance is not fitted yet"
        message = "Expected NotFittedError regarding fitting"

        with pytest.raises(NotFittedError, match=match, message=message):
            self.api_call(DecisionTreeClassifier(), *self.min_args)


class TestGetTreeInfo(BaseApiTest):

    min_args = ()

    @staticmethod
    def api_call(*args, **kwargs):
        return api.get_tree_info(*args, **kwargs)


class TestGetDecisionInfo(BaseApiTest):

    min_args = (np.array([]),)

    @staticmethod
    def api_call(*args, **kwargs):
        return api.get_decision_info(*args, **kwargs)
