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


@pytest.mark.parametrize("tab_size", [-5, 0, 2, 5, None])
def test_get_tab(tab_size):
    tab_size = 5 if tab_size is None else max(0, tab_size)
    assert utils.get_tab(tab_size) == " " * tab_size
