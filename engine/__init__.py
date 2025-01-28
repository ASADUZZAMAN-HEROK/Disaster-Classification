from .train_engine import Engine


def build_engine(engine_name: str):
    if engine_name == "engine":
        return Engine
    else:
        raise ValueError(f"Unknown engine: {engine_name}")
