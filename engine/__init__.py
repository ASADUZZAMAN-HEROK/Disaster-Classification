from .train_engine import Engine
from .test_engine import TestEngine


def build_engine(engine_name: str):
    if engine_name == "engine":
        return Engine
    if engine_name == "test_engine":
        return TestEngine
    else:
        raise ValueError(f"Unknown engine: {engine_name}")
