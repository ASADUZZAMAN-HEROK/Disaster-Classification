from .kfold_train_engin import KFoldEngine


def build_engine(engine_name: str):
    if engine_name == "kfold_train_engine":
        return KFoldEngine
    else:
        raise ValueError(f"Unknown engine: {engine_name}")
