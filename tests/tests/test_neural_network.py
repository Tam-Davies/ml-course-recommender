import pytest

tf = pytest.importorskip("tensorflow")

from src.models.neural_network import NeuralNetworkRecommender


def test_neural_network_model_shapes():
    recommender = NeuralNetworkRecommender(
        n_users=5,
        n_items=3,
        embedding_dim=4,
    )

    model = recommender.build_model()

    assert model.input_shape == [(None, 1), (None, 1)]
    assert model.output_shape == (None,)


def test_neural_network_parameter_count():
    recommender = NeuralNetworkRecommender(
        n_users=5,
        n_items=3,
        embedding_dim=4,
    )

    model = recommender.build_model()

    expected_params = (5 * 4) + (3 * 4)

    assert model.count_params() == expected_params
