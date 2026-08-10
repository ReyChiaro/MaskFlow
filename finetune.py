import hydra
from omegaconf import OmegaConf
from hydra.utils import instantiate
from trainer.base_trainer import BaseTrainer
from loguru import logger
from utils.config import register_distributed_timestamp_resolver

register_distributed_timestamp_resolver()


@hydra.main(config_path="configs", config_name="train", version_base="v1.2")
def finetune(cfgs: OmegaConf):
    cfg_contents = "\n" + " Configs ".center(50, "=")
    cfg_contents += "\n" + OmegaConf.to_yaml(cfgs)
    cfg_contents += "\n" + "=" * 50
    logger.info(cfg_contents)

    trainer: BaseTrainer = instantiate(cfgs.trainer)
    trainer.train()


if __name__ == "__main__":
    finetune()
