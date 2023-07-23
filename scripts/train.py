from pathlib import Path

import logging
import pytorch_lightning as pl
import hydra

from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from cross_view_transformer.common import setup_config, setup_experiment, load_backbone, load_backbone_road, load_backbone_vehicle 
from cross_view_transformer.callbacks.gitdiff_callback import GitDiffCallback
from cross_view_transformer.callbacks.visualization_callback import VisualizationCallback

log = logging.getLogger(__name__)

CONFIG_PATH = Path.cwd() / 'config'
CONFIG_NAME = 'config.yaml'


def maybe_resume_training(experiment):
    save_dir = Path(experiment.save_dir).resolve()
    checkpoints = None

    # checkpoints = list(save_dir.glob(f'**/{experiment.uuid}/checkpoints/*.ckpt'))
    # checkpoints = list(save_dir.glob(f'**/cvt_nuscenes_vehicles_50k.ckpt'))

    checkpoints = list(save_dir.glob(f'**/0714_170410/checkpoints/*.ckpt'))

    
    log.info(f'Searching {save_dir}.')

    if not checkpoints:
        return None

    log.info(f'Found {checkpoints[-1]}.')

    return checkpoints[-1]


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):
    setup_config(cfg)

    pl.seed_everything(cfg.experiment.seed, workers=True)

    Path(cfg.experiment.save_dir).mkdir(exist_ok=True, parents=False)

    # Create and load model/data
    model_module, data_module, viz_fn = setup_experiment(cfg)

    # seperate model loading for cross_view_transformers_waypoint
    if cfg.experiment.project == 'cross_view_transformers_waypoint':
        road_ckpt_path = list(Path(cfg.experiment.save_dir).glob('**/road_ckpt/*.ckpt'))[0]
        log.info(f'Found {road_ckpt_path}.')

        vehicle_ckpt_path = list(Path(cfg.experiment.save_dir).glob('**/vehicle_ckpt/*.ckpt'))[0]
        log.info(f'Found {vehicle_ckpt_path}.')

        model_module.backbone.cvt_road_encoder = load_backbone_road(road_ckpt_path)
        log.info(f'Loaded {road_ckpt_path}.')

        model_module.backbone.cvt_vehicle_encoder = load_backbone_vehicle(vehicle_ckpt_path)
        log.info(f'Loaded {vehicle_ckpt_path}.')

    else:
        # Optionally load model
        ckpt_path = maybe_resume_training(cfg.experiment)

        if ckpt_path is not None:
            model_module.backbone = load_backbone(ckpt_path)

    # Loggers and callbacks
    logger = pl.loggers.WandbLogger(project=cfg.experiment.project,
                                    save_dir=cfg.experiment.save_dir,
                                    id=cfg.experiment.uuid)

    # seperate callbacks for cross_view_transformers_waypoint
    if cfg.experiment.project == 'cross_view_transformers_waypoint':
        callbacks = [
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(filename='model',
                            every_n_train_steps=cfg.experiment.checkpoint_interval),
            # VisualizationCallback(viz_fn, cfg.experiment.log_image_interval), # no visualization for waypoint
            GitDiffCallback(cfg)
        ]
    else:
        callbacks = [
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(filename='model',
                            every_n_train_steps=cfg.experiment.checkpoint_interval),
            VisualizationCallback(viz_fn, cfg.experiment.log_image_interval),
            GitDiffCallback(cfg)
        ]

    # Train
    trainer = pl.Trainer(logger=logger,
                         callbacks=callbacks,
                        #  strategy=DDPStrategy(find_unused_parameters=False),
                         accelerator="gpu",
                         **cfg.trainer,
                         fast_dev_run=False)
    
    ckpt_path = None

    trainer.fit(model_module, datamodule=data_module, ckpt_path=ckpt_path)

if __name__ == '__main__':
    main()
