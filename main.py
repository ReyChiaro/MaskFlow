import hydra

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration
from omegaconf import OmegaConf
from hydra.utils import instantiate

# from trainer.trainer import Trainer
from trainer import BaseTrainer
from loguru import logger


@hydra.main(config_path="configs", config_name="train", version_base="v1.2")
def main(cfgs: OmegaConf):
    cfg_contents = "\n" + " Configs ".center(50, "=")
    cfg_contents += "\n" + OmegaConf.to_yaml(cfgs)
    cfg_contents += "\n" + "=" * 50
    logger.info(cfg_contents)

    # accelerator = Accelerator(
    #     gradient_accumulation_steps=cfgs.gradient_accumulation_steps,
    #     log_with=cfgs.log_with,
    #     project_config=ProjectConfiguration(project_dir=cfgs.output_dir, logging_dir=cfgs.log_dir),
    #     kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=cfgs.enable_find_unused_parameters)],
    # )

    trainer: BaseTrainer = instantiate(cfgs.trainer)
    trainer.train()


if __name__ == "__main__":
    main()
