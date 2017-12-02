"""
Useful utilities for our tree-decoding API.
"""

from sklearn.tree import DecisionTreeClassifier


def check_estimator_type(estimator):
    """
    Check that the data type of estimator is one that we support.

    Currently, we only support `sklearn.tree.DecisionTreeClassifier`,
    though support for other trees is forthcoming.

    Parameters
    ----------
    estimator : object
        The estimator to check.

    Raises
    ------
    NotImplementedError : the data type of the estimator was one that
                          we do not support at the moment.
    """

    if not isinstance(estimator, DecisionTreeClassifier):
        raise NotImplementedError("Function support is only implemented for "
                                  "DecisionTreeClassifier. Support for "
                                  "other trees is forthcoming.")


def maybe_round(val, precision=None):
    """
    Potentially round a number or an array of numbers to a given precision.

    Parameters
    ----------
    val : numeric or numpy.ndarray
        The number or array of numbers to round.
    precision : int, default None
        The precision at which to perform rounding. If None is provided,


    Returns
    -------
    maybe_rounded_val : the number or array of numbers rounded to a
                        given precision, if provided. Otherwise, the
                        original input is returned.
    """

    if precision is None:
        return val

    if hasattr(val, "round"):
        return val.round(precision)
    else:
        return round(val, precision)
