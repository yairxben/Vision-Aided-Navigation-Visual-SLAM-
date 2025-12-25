import numpy as np

DET = "Determinant"
EQUAL = "EQUAL"
WEIGHT_METHOD = DET


def cov_weight(cov, weight_method=DET):
    """
    Return cov weighting
    :param cov: Covariance between two cameras
    """
    if cov is None:
        return 0

    if weight_method is DET:
        return np.sqrt(np.linalg.det(cov))

    if weight_method is EQUAL:
        return 1


class Edge:

    def __init__(self, source, target, cov, weighted_det=True):
        """
        :param source: First vertex
        :param target: Second vertex
        :param cov: Covariance matrix
        :param weighted_det: If True, the edge is weighted; otherwise, it is not.
        """
        self.source = source
        self.target = target
        self.__cov = cov
        if weighted_det:
            self.__weight = np.sqrt(np.linalg.det(cov))
        else:
            self.__weight = 1

    def get_cov(self):
        return self.__cov

    def get_weight(self):
        return self.__weight

    def get_vertices(self):
        return self.source, self.target

    def __eq__(self, other):
        return self.source == other.source and self.target == other.target

    def __hash__(self):
        return hash((self.source, self.target))