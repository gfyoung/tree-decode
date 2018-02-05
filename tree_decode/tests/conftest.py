from sklearn.tree import (DecisionTreeClassifier, DecisionTreeRegressor,
                          ExtraTreeClassifier, ExtraTreeRegressor)

import pytest


_SUPPORTED = (DecisionTreeClassifier, DecisionTreeRegressor,
              ExtraTreeClassifier, ExtraTreeRegressor)


@pytest.fixture(params=_SUPPORTED)
def tree(request):
    """
    Fixture all supported decision tree classes and ensembles to test.
    """

    return request.param
