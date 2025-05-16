from src.enums import Feature, PlayHand

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

def testPlayHandSubstraction():
    """
    Test the method PlayHand.__sub__
    """
    assert PlayHand.LEFT - PlayHand.LEFT == 0
    assert PlayHand.RIGHT - PlayHand.RIGHT == 0
    assert PlayHand.RIGHT - PlayHand.LEFT == 1
    assert PlayHand.LEFT - PlayHand.RIGHT == -1
