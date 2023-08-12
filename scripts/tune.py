import sys
sys.path.insert(0, '.')

import os
# # Set the wandb directory to the current working directory
# os.environ['WANDB_DIR'] = os.getcwd()
# os.environ['WANDB_CACHE_DIR '] = os.getcwd()
# os.environ['WANDB_CONFIG_DIR'] = os.getcwd()

import wandb
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.integration.wandb import WeightsAndBiasesCallback
# from optuna.visualization import plot_contour
# from optuna.visualization import plot_edf
# from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
# from optuna.visualization import plot_rank
# from optuna.visualization import plot_slice
# from optuna.visualization import plot_timeline

from pathlib import Path

import logging
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import hydra

from pytorch_lightning import LightningModule
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from cross_view_transformer.common import setup_config, setup_experiment, load_backbone, load_backbone_road, load_backbone_vehicle, load_backbone_lane
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

    # checkpoints = list(save_dir.glob(f'**/0731_173821/checkpoints/*.ckpt'))
    
    log.info(f'Searching {save_dir}.')

    if not checkpoints:
        return None

    log.info(f'Found {checkpoints[-1]}.')

    return checkpoints[-1]

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):

    def objective(cfg, trial):
        
        # Define the search space and sample hyperparameters
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        # batch_size = trial.suggest_categorical("batch_size", [4])
        # num_layers = trial.suggest_categorical("num_layers", [1, 2])
        # model_heads = trial.suggest_categorical("model_heads", [4, 8])
        model_heads = 8
        # model_dim = trial.suggest_categorical("model_dim", [128, 256])
        model_dim = 256
        # num_SAs = trial.suggest_categorical("num_SAs", [1, 2, 3])

        # Update Hydra config with the sampled hyperparameters
        cfg.optimizer.lr = lr
        # cfg.loader.batch_size = batch_size
        # cfg.model.num_layers = num_layers
        cfg.model.heads = model_heads
        cfg.model.dim_head = int(model_dim / model_heads)
        # cfg.model.num_SAs = num_SAs

        # update scheduler.steps_per_epoch
        cfg.scheduler.max_lr = lr
        
        # HACK: hardcode steps_per_epoch
        cfg.scheduler.steps_per_epoch = cfg.data.train.num_samples // cfg.loader.batch_size + 1

        # assert cfg.scheduler.steps_per_epoch is not None
        assert cfg.scheduler.steps_per_epoch is not None, 'specify steps_per_epoch in scheduler'

        setup_config(cfg)
        
        pl.seed_everything(cfg.experiment.seed, workers=True)

        Path(cfg.experiment.save_dir).mkdir(exist_ok=True, parents=False)

        # Create and load model/data
        model_module, data_module, viz_fn = setup_experiment(cfg)

        ckpt_path = maybe_resume_training(cfg.experiment)

        if ckpt_path is not None:
            log.info(f'Found {ckpt_path}.')
            model_module.backbone = load_backbone(ckpt_path)
            log.info(f'Loaded {ckpt_path}.')

        # seperate model loading for cross_view_transformers_waypoint
        if 'cross_view_transformers_waypoint' in cfg.experiment.project:
            lane_ckpt_path = list(Path(cfg.experiment.save_dir).glob('**/lane_ckpt/*.ckpt'))[0]
            log.info(f'Found {lane_ckpt_path}.')

            road_ckpt_path = list(Path(cfg.experiment.save_dir).glob('**/road_ckpt/*.ckpt'))[0]
            log.info(f'Found {road_ckpt_path}.')

            vehicle_ckpt_path = list(Path(cfg.experiment.save_dir).glob('**/vehicle_ckpt/*.ckpt'))[0]
            log.info(f'Found {vehicle_ckpt_path}.')

            if model_module.backbone.cvt_lane_encoder:
                model_module.backbone.cvt_lane_encoder = load_backbone_lane(lane_ckpt_path)
                log.info(f'Loaded {lane_ckpt_path}.')

            if model_module.backbone.cvt_road_encoder:
                model_module.backbone.cvt_road_encoder = load_backbone_road(road_ckpt_path)
                log.info(f'Loaded {road_ckpt_path}.')

            if model_module.backbone.cvt_vehicle_encoder:
                model_module.backbone.cvt_vehicle_encoder = load_backbone_vehicle(vehicle_ckpt_path)
                log.info(f'Loaded {vehicle_ckpt_path}.')

        else:
            # Optionally load model
            ckpt_path = maybe_resume_training(cfg.experiment)

            if ckpt_path is not None:
                model_module.backbone = load_backbone(ckpt_path)

        # # Loggers and callbacks
        logger = pl.loggers.WandbLogger(project=cfg.experiment.project,
                                        save_dir=cfg.experiment.save_dir,
                                        id=cfg.experiment.uuid + "_trial" + str(trial.number),
                                        )

        callbacks = [
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(filename='model',
                            every_n_train_steps=cfg.experiment.checkpoint_interval),
            VisualizationCallback(viz_fn, cfg.experiment.log_image_interval),
            GitDiffCallback(cfg)
        ]

        # Add the Optuna pruning callback
        callbacks.append(PyTorchLightningPruningCallback(trial, monitor='val/metrics/ADE'))

        trainer = pl.Trainer(
            logger=logger,
            callbacks=callbacks,
            accelerator="gpu",
            **cfg.trainer,
            fast_dev_run=False
        )

        trainer.fit(model_module, datamodule=data_module, ckpt_path=ckpt_path)
        wandb.finish()
        
        # Return the metric you want to optimize
        # return trainer.callback_metrics['train/loss'].item() # for debug
        return trainer.callback_metrics['val/metrics/ADE'].item()

    if __name__ == '__main__':
        study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())

        study.optimize(lambda trial: objective(cfg, trial), n_trials=30)
        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        # send optuna results to wandb
        fig1 = plot_parallel_coordinate(study)
        fig2 = plot_optimization_history(study)
        fig3 = plot_param_importances(study)

        fig1_path = "parallel_coordinate.html"
        fig2_path = "optimization_history.html"
        fig3_path = "param_importances.html"

        fig1.write_html(fig1_path)
        fig2.write_html(fig2_path)
        fig3.write_html(fig3_path)

        table = wandb.Table(columns = ["parallel_coordinate", "optimization_history", "param_importances"])
        table.add_data(wandb.Html(fig1_path), wandb.Html(fig2_path), wandb.Html(fig3_path))

        wandb.init(project=cfg.experiment.project, name=cfg.experiment.uuid + "_optuna_results")
        wandb.log({"optuna results": table})
        wandb.finish()


if __name__ == '__main__':
    main()