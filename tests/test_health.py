import os

def test_env_loaded():
    assert os.getenv("DUMMY_VARIABLE") is not None