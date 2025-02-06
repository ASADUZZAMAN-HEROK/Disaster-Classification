import accelerate
import tyro
import argparse

from configs import Config
from engine import build_engine


def main():
    cfg = tyro.cli(Config)

    if cfg.config is not None:
        with open(cfg.config, "r") as f:
            json_cfg = Config.from_json(f.read())
        if cfg.model.resume_path is not None:
            json_cfg.model.resume_path = cfg.model.resume_path
        if cfg.model.weight_path is not None:
            json_cfg.model.weight_path = cfg.model.weight_path
        
        json_cfg.model.pretrained = cfg.model.pretrained
        json_cfg.training.epochs = cfg.training.epochs
        cfg = json_cfg 

    project_config = accelerate.utils.ProjectConfiguration(
        project_dir=cfg.project_dir,
        logging_dir=cfg.log_dir,
    )
    accelerator = accelerate.Accelerator(
        log_with=cfg.project_tracker,
        project_config=project_config,
        gradient_accumulation_steps=cfg.training.accum_iter,
        mixed_precision=cfg.mixed_precision,
    )

    accelerate.utils.set_seed(cfg.seed, device_specific=True)
    engine = build_engine(cfg.training.engine)(accelerator, cfg)
    engine.train()
    engine.close()


if __name__ == "__main__":
    # exit(0) #Accidental retrain
    main()
