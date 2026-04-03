from src.data import IrisData
from src.predict import predict_species_with_confidence


def test_predict_returns_valid_output():
    sample = IrisData(
        sepal_length=5.1,
        sepal_width=3.5,
        petal_length=1.4,
        petal_width=0.2
    )
    pred, conf = predict_species_with_confidence(sample)

    assert pred in [0, 1, 2]
    assert 0.0 <= conf <= 1.0
