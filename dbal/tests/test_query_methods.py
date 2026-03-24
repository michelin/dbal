import os
import sys

import numpy as np
import pytest
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

from dbal.query_methods import (
    BaseQuery,
    DiversityQuery,
    KCentersQuery,
    KMeansQuery,
    KMedoidsAccelerated,
    KMedoidsQuery,
    OrderedQuery,
    RandomQuery,
)
from dbal.utils import toy_example


@pytest.fixture
def basic_data():
    """Fixture providing basic test data using toy_example"""
    np.random.seed(42)
    Xs, Xt, f = toy_example(size=50, dim=2, cluster=3)
    ys = np.random.randint(0, 2, len(Xs))
    sample_weight = np.ones(len(Xt))
    return Xs, Xt[:40], ys, sample_weight[:40]


@pytest.fixture
def small_data():
    """Fixture providing small test data for quick tests"""
    np.random.seed(42)
    Xs, Xt, f = toy_example(size=20, dim=2, cluster=2)
    ys = np.random.randint(0, 2, len(Xs))
    sample_weight = np.ones(len(Xt))
    return Xs, Xt, ys, sample_weight


class TestOrderedQuery:
    """Basic tests for OrderedQuery"""

    def test_basic_functionality(self, basic_data):
        """Test OrderedQuery basic functionality"""
        Xs, Xt, ys, sample_weight = basic_data
        # Create non-uniform weights for testing
        sample_weight = np.random.random(len(Xt))

        query = OrderedQuery()
        query.fit(Xt, sample_weight=sample_weight, n_queries=5)

        result = query.predict(n_queries=5)
        assert len(result) == 5
        assert all(isinstance(idx, (int, np.integer)) for idx in result)

        # Should return indices in descending order of weights
        weights_of_selected = sample_weight[result]
        for i in range(len(weights_of_selected) - 1):
            assert weights_of_selected[i] >= weights_of_selected[i + 1]

    def test_without_weights(self, small_data):
        """Test OrderedQuery without explicit weights"""
        Xs, Xt, ys, sample_weight = small_data

        query = OrderedQuery()
        query.fit(Xt, n_queries=3)

        result = query.predict(n_queries=3)
        assert len(result) == 3


class TestKMedoidsQuery:
    """Basic tests for KMedoidsQuery"""

    def test_basic_functionality(self, basic_data):
        """Test KMedoidsQuery basic functionality"""
        Xs, Xt, ys, sample_weight = basic_data

        query = KMedoidsQuery(distance=manhattan_distances, nn_algorithm="brute")
        query.fit(Xt, Xs, ys, n_queries=4, sample_weight=sample_weight)

        result = query.predict(n_queries=4)
        assert len(result) == 4
        assert all(isinstance(idx, (int, np.integer)) for idx in result)
        assert all(0 <= idx < len(Xt) for idx in result)

        # Check that deltas are computed
        assert hasattr(query, "deltas")
        assert len(query.deltas) == len(Xt)

    def test_without_source_data(self, small_data):
        """Test KMedoidsQuery without source data"""
        Xs, Xt, ys, sample_weight = small_data

        query = KMedoidsQuery()
        query.fit(Xt, Xs=None, n_queries=3, sample_weight=sample_weight)

        result = query.predict(n_queries=3)
        assert len(result) == 3

    def test_euclidean_distance(self, small_data):
        """Test KMedoidsQuery with Euclidean distance"""
        Xs, Xt, ys, sample_weight = small_data

        query = KMedoidsQuery(distance=euclidean_distances)
        query.fit(Xt, Xs, ys, n_queries=2, sample_weight=sample_weight)

        result = query.predict(n_queries=2)
        assert len(result) == 2

    def test_kd_trees_algorithm(self, small_data):
        """Test KMedoidsQuery with KD-trees algorithm"""
        Xs, Xt, ys, sample_weight = small_data

        query = KMedoidsQuery(nn_algorithm="kd-trees")
        query.fit(Xt, Xs, ys, n_queries=2, sample_weight=sample_weight)

        result = query.predict(n_queries=2)
        assert len(result) == 2

    def test_invalid_algorithm(self, small_data):
        """Test KMedoidsQuery with invalid algorithm"""
        Xs, Xt, ys, sample_weight = small_data

        query = KMedoidsQuery(nn_algorithm="invalid")

        with pytest.raises(ValueError, match="nn_algorithm should be"):
            query.fit(Xt, Xs, ys, n_queries=2, sample_weight=sample_weight)


class TestKMedoidsAccelerated:
    """Basic tests for KMedoidsAccelerated"""

    def test_basic_functionality(self, basic_data):
        """Test KMedoidsAccelerated basic functionality"""
        Xs, Xt, ys, sample_weight = basic_data

        query = KMedoidsAccelerated(batch_size_init=20, max_iter=2, verbose=0)
        query.fit(Xt, Xs, ys, n_queries=3, sample_weight=sample_weight)

        result = query.predict(n_queries=3)
        assert len(result) == 3
        assert all(isinstance(idx, (int, np.integer)) for idx in result)
        assert all(0 <= idx < len(Xt) for idx in result)

    def test_without_source_data(self, small_data):
        """Test KMedoidsAccelerated without source data"""
        Xs, Xt, ys, sample_weight = small_data

        query = KMedoidsAccelerated(batch_size_init=10, max_iter=1)
        query.fit(Xt, Xs=None, n_queries=2, sample_weight=sample_weight)

        result = query.predict(n_queries=2)
        assert len(result) == 2

    def test_kdt_forest_algorithm(self, small_data):
        """Test KMedoidsAccelerated with KDT-forest"""
        Xs, Xt, ys, sample_weight = small_data

        query = KMedoidsAccelerated(
            nn_algorithm="kdt-forest", batch_size_init=15, max_iter=1
        )
        query.fit(Xt, Xs, ys, n_queries=2, sample_weight=sample_weight)

        result = query.predict(n_queries=2)
        assert len(result) == 2


class TestKMeansQuery:
    """Basic tests for KMeansQuery"""

    def test_basic_functionality(self, basic_data):
        """Test KMeansQuery basic functionality"""
        Xs, Xt, ys, sample_weight = basic_data

        query = KMeansQuery(minibatch=False)
        query.fit(Xt, sample_weight=sample_weight, n_queries=4)

        result = query.predict(n_queries=4)
        assert len(result) == 4
        assert all(isinstance(idx, (int, np.integer)) for idx in result)
        assert all(0 <= idx < len(Xt) for idx in result)
        # Should return unique indices
        assert len(set(result)) == len(result)

    def test_minibatch_kmeans(self, small_data):
        """Test KMeansQuery with MiniBatch"""
        Xs, Xt, ys, sample_weight = small_data

        query = KMeansQuery(minibatch=True)
        query.fit(Xt, sample_weight=sample_weight, n_queries=3)

        result = query.predict(n_queries=3)
        assert len(result) == 3


class TestRandomQuery:
    """Basic tests for RandomQuery"""

    def test_basic_functionality(self, basic_data):
        """Test RandomQuery basic functionality"""
        Xs, Xt, ys, sample_weight = basic_data

        query = RandomQuery()
        query.fit(Xt, sample_weight=sample_weight, n_queries=5)

        result = query.predict(n_queries=5)
        assert len(result) == 5
        assert all(isinstance(idx, (int, np.integer)) for idx in result)
        assert all(0 <= idx < len(Xt) for idx in result)
        # Should return unique indices (replace=False)
        assert len(set(result)) == len(result)


class TestKCentersQuery:
    """Basic tests for KCentersQuery"""

    def test_basic_functionality(self, basic_data):
        """Test KCentersQuery basic functionality"""
        Xs, Xt, ys, sample_weight = basic_data

        query = KCentersQuery(distance=manhattan_distances, nn_algorithm="brute")
        query.fit(Xt, Xs, ys, n_queries=4, sample_weight=sample_weight)

        result = query.predict(n_queries=4)
        assert len(result) == 4
        assert all(isinstance(idx, (int, np.integer)) for idx in result)
        assert all(0 <= idx < len(Xt) for idx in result)
        # Should return unique indices
        assert len(set(result)) == len(result)

    def test_without_source_data(self, small_data):
        """Test KCentersQuery without source data"""
        Xs, Xt, ys, sample_weight = small_data

        query = KCentersQuery()
        query.fit(Xt, Xs=None, n_queries=3, sample_weight=sample_weight)

        result = query.predict(n_queries=3)
        assert len(result) == 3

    def test_without_weights(self, small_data):
        """Test KCentersQuery without sample weights"""
        Xs, Xt, ys, sample_weight = small_data

        query = KCentersQuery()
        query.fit(Xt, Xs, ys, n_queries=2)

        result = query.predict(n_queries=2)
        assert len(result) == 2

    def test_kdt_forest_algorithm(self, small_data):
        """Test KCentersQuery with KDT-forest"""
        Xs, Xt, ys, sample_weight = small_data

        query = KCentersQuery(nn_algorithm="kdt-forest")
        query.fit(Xt, Xs, ys, n_queries=2, sample_weight=sample_weight)

        result = query.predict(n_queries=2)
        assert len(result) == 2

    def test_invalid_algorithm(self, small_data):
        """Test KCentersQuery with invalid algorithm"""
        Xs, Xt, ys, sample_weight = small_data

        query = KCentersQuery(nn_algorithm="invalid")

        with pytest.raises(ValueError, match="nn_algorithm should be"):
            query.fit(Xt, Xs, ys, n_queries=2, sample_weight=sample_weight)


class TestDiversityQuery:
    """Basic tests for DiversityQuery"""

    def test_basic_functionality(self, basic_data):
        """Test DiversityQuery basic functionality"""
        Xs, Xt, ys, sample_weight = basic_data

        query = DiversityQuery(distance=manhattan_distances, nn_algorithm="brute")
        query.fit(Xt, Xs, ys, n_queries=5, sample_weight=sample_weight)

        result = query.predict(n_queries=5)
        assert len(result) == 5
        assert all(isinstance(idx, (int, np.integer)) for idx in result)
        assert all(0 <= idx < len(Xt) for idx in result)

    def test_without_source_data(self, small_data):
        """Test DiversityQuery without source data"""
        Xs, Xt, ys, sample_weight = small_data

        query = DiversityQuery()
        query.fit(Xt, Xs=None, n_queries=3, sample_weight=sample_weight)

        result = query.predict(n_queries=3)
        assert len(result) == 3

    def test_kdt_forest_algorithm(self, small_data):
        """Test DiversityQuery with KDT-forest"""
        Xs, Xt, ys, sample_weight = small_data

        query = DiversityQuery(nn_algorithm="kdt-forest")
        query.fit(Xt, Xs, ys, n_queries=3, sample_weight=sample_weight)

        result = query.predict(n_queries=3)
        assert len(result) == 3
