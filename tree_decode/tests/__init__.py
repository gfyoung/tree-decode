"""
Top-level directory for all of tree-decode's tests.
"""

from sklearn.tree import (DecisionTreeClassifier, DecisionTreeRegressor,
                          ExtraTreeClassifier, ExtraTreeRegressor)
from tree_decode.tests.utils import *  # noqa

_SUPPORTED = (DecisionTreeClassifier, DecisionTreeRegressor,
              ExtraTreeClassifier, ExtraTreeRegressor)
