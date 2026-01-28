from omegaconf import OmegaConf, DictConfig
from runner.backbone_trainer import BackboneTrainer
from utils import seed_everything

def run(conf: DictConfig) -> None:
    exp = BackboneTrainer(conf=conf)
    exp.start_training()


if __name__ == "__main__":
    seed_everything()
    conf = OmegaConf.load("config/backbone.yaml")
    run(conf)