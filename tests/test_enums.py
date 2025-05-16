from src.enums import Feature

def test_get_features_by_type():
    """
    Test the method Feature.get_features_by_type
    """
    features = Feature.get_features_by_type('category')
    assert len(features) == 3
    assert all([feature.type == 'category' for feature in features])

    features = Feature.get_features_by_type('number')
    assert len(features) == 10
    assert all([feature.type == 'number' for feature in features])

def test_get_all_features():
    """
    Test the method Feature.get_all_features
    """
    features = Feature.get_all_features()
    assert len(features) == 13

