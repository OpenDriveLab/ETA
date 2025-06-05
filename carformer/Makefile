# By default echo the help message
.DEFAULT_GOAL := help

help: ## Display this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_-]+:.*?## / {printf " %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

ETA_base_model_s42:
	OMP_NUM_THREADS=6 python train.py deepspeed=zero2 hyperparams.batch_size=5 hyperparams.num_epochs=40 user=example wandb_tag=$@ dataset.dataset_path_rel=B2D-base gpus=4 nodes=8 training.parallel_dataset_init=False backbone=llama-base training.weighted_sampling=True training/goal=dual_target_point training.normalize_goal=True training.bucket_weights.type=preferturns training.rgb_crop.crop_size=896 start_saving_epoch=30 seed=42

ETA_async_model_s42: 
	OMP_NUM_THREADS=6 python train.py deepspeed=zero2 hyperparams.batch_size=5 hyperparams.num_epochs=40 user=example wandb_tag=$@ dataset.dataset_path_rel=B2D-base gpus=4 nodes=8 training.parallel_dataset_init=False training.weighted_sampling=True training/goal=dual_target_point training.normalize_goal=True training.bucket_weights.type=preferturns training.use_future_vehicle_forecast=True training.use_predicted_latent_with_gap=True training.action_gap=1 training.future_horizon=1 training.ema_enabled=False +experiments=[use_light_encoder] training.ignore_past_for_length=True training.rgb_crop.crop_size=896 seed=42
