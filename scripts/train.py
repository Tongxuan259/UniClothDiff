import os
import math
import shutil
import logging
import importlib
import numpy as np

import torch
from torch.utils.data import RandomSampler
import transformers

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import DistributedDataParallelKwargs

import diffusers
from diffusers.optimization import get_scheduler
import wandb
from args import parse_args
from uniclothdiff.registry import (
    build_dataset, 
    build_model, 
    build_scheduler
)
from uniclothdiff.utils.torch_utils import to_torch_dtype
from uniclothdiff.utils.training_utils import (
    get_model_numel, 
    format_numel_str, 
    backup_code, 
    find_unused_parameters,
    get_model_parameters
)
from uniclothdiff.pipelines.cloth_dynamics_pipeline import ClothDynamicsPipeline
from tqdm.auto import tqdm
from tqdm import tqdm
from omegaconf import OmegaConf


logger = get_logger(__name__, log_level="INFO")

def setup_accelerator(project_dir: str, 
                      logging_dir: str, 
                      gradient_accumulation_steps: int,
                      mixed_precision: str,
                      logs_report_to: str,
                      ):
    accelerator_project_config = ProjectConfiguration(
        project_dir=project_dir, logging_dir=logging_dir)
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=logs_report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs]
    )
    
    return accelerator

def main():
    assert torch.cuda.is_available(), \
        "Training requires at least one GPU."
        
    args = parse_args()
    args_dict = vars(args)
    exp_cfg = OmegaConf.load(args.config)
    config = OmegaConf.create(args_dict)
    config = OmegaConf.merge(config, exp_cfg)
    config.dataset_cfg.data_dir = args.data_dir 
    # Handle the repository creation
    experiment_dir = os.path.join("experiments", f"{config.exp_name}")
    acc_logging_dir = os.path.join(experiment_dir, config.logging_dir)
    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    
    accelerator = setup_accelerator(
        experiment_dir, 
        acc_logging_dir, 
        config.gradient_accumulation_steps,
        config.mixed_precision,
        config.report_to
    )
    
    generator = torch.Generator(device=accelerator.device).manual_seed(config.seed)
    
    if accelerator.is_main_process:
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)
    
    if accelerator.is_main_process and config.exp_name != "DEBUG":
        backup_code(experiment_dir, logger=logger)
        OmegaConf.save(config, os.path.join(experiment_dir, "config.yml"))
        wandb.init(
            project=config.wandb_cfg.project_name,
            entity=config.wandb_cfg.entity,
            tags=config.wandb_cfg.tags,
            name=config.exp_name
        )
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    
    set_seed(config.seed)
    
    diffusion_scheduler = build_scheduler(OmegaConf.to_container(config.diffusion_cfg))
    
    if config.pretrained_model_name_or_path:
        model_cls = getattr(importlib.import_module("uniclothdiff.models"), config.model_cfg.type)
        model = model_cls.from_pretrained(
            config.pretrained_model_name_or_path,
            subfolder="model",
            low_cpu_mem_usage=False,
        )
    else:
        model = build_model(OmegaConf.to_container(config.model_cfg))
        
    model_numel, model_numel_trainable = get_model_numel(model)
    # update_params_name, update_params= get_model_parameters(model)        # uncomment it to find unused params
    
    cfg_dtype = config.get("mixed_precision", "float32")
    weight_dtype = to_torch_dtype(cfg_dtype)

    model.requires_grad_(False)
    
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "model"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(models, input_dir):
        for i in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()
            
            class_name = type(model).__name__  # get object class name
            module_name = model.__module__  # get the module where class is in
            module = importlib.import_module(module_name)  # import module
            ModelClass = getattr(module, class_name)  # import class
            
            # load diffusers style into model
            load_model = ModelClass.from_pretrained(input_dir, subfolder="model")
            model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())

            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    
    if config.gradient_checkpointing:
        model.enable_gradient_checkpointing() 
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if config.scale_lr:
        config.learning_rate = (
            config.learning_rate * config.gradient_accumulation_steps *
            config.per_gpu_batch_size * accelerator.num_processes
        )
    
    optimizer_cls = torch.optim.AdamW
    model.requires_grad_(True)
    


    optimizer = optimizer_cls(
        model.parameters(),
        # update_params,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )
    
    # DataLoaders creation:
    config.global_batch_size = config.per_gpu_batch_size * accelerator.num_processes
    
    train_dataset_cfg = OmegaConf.to_container(config.dataset_cfg)
    train_dataset_cfg["mode"] = "train"
    valid_dataset_cfg = OmegaConf.to_container(config.dataset_cfg)
    valid_dataset_cfg["mode"] = "valid"
    train_dataset = build_dataset(train_dataset_cfg)
    valid_dataset = build_dataset(valid_dataset_cfg)
    
    data_sampler = RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=data_sampler,
        batch_size=config.per_gpu_batch_size,
        num_workers=config.num_workers,
        pin_memory=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.per_gpu_batch_size,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config.max_train_steps * accelerator.num_processes,
    )
    
    # Prepare everything with our `accelerator`.
    model, optimizer, lr_scheduler, train_dataloader, valid_dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader, valid_dataloader
    )
    
    # length of dataloader may change after prepared with accelerator
    num_update_step_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    config.max_train_steps = config.num_train_epochs * num_update_step_per_epoch
    config.num_train_epochs = math.ceil(config.max_train_steps / num_update_step_per_epoch)
    
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="ClothDiffusion", 
            config=OmegaConf.to_container(config, resolve=True)
        )

    # Train!
    total_batch_size = config.per_gpu_batch_size * accelerator.num_processes * config.gradient_accumulation_steps
    
    logger.info("******** Running training ********")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {config.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.per_gpu_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")
    logger.info("  Total model prarameter = %s", format_numel_str(model_numel),)
    logger.info("  Trainable model prarameter = %s", format_numel_str(model_numel_trainable),)
    logger.info(f"  Do classifier free guidance = {config.do_classifier_free_guidance}")
    
    global_step = 0
    first_epoch = 0
    
    # Potentially load in the weights and states from a previous save
    if config.resume_from_checkpoint:
        if config.resume_from_checkpoint != "latest":
            # path = os.path.basename(args.resume_from_checkpoint)
            path = config.resume_from_checkpoint
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(checkpoints_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{config.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            config.resume_from_checkpoint = None
        else:
            accelerator.print(f"Loading state from checkpoint {path}")
            if os.path.exists(path):
                accelerator.print(f"Fine-tuning with previous experiments checkpoint")
                accelerator.load_state(path)
                global_step = 0
            else:
                accelerator.print(f"Resuming from previous experiments")
                accelerator.load_state(os.path.join(checkpoints_dir, path))
                global_step = int(path.split("-")[1])

            resume_global_step = global_step * config.gradient_accumulation_steps
            first_epoch = global_step // num_update_step_per_epoch
            resume_step = resume_global_step % (num_update_step_per_epoch * config.gradient_accumulation_steps)

    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, config.max_train_steps),
        disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")
    for epoch in range(first_epoch, config.num_train_epochs):
        model.train()
        train_loss = 0.0
        valid_loss = 0.0

        # ******** training loop in an epoch ******** #
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if config.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % config.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(model):
                input = batch.pop('q_delta')        # this is for training dynamics
                if not config.do_classifier_free_guidance:
                    loss = diffusion_scheduler.training_losses(
                        model=model, 
                        input=input,
                        model_kwargs=batch,
                        weight_dtype=weight_dtype
                    )
                else:
                    loss = diffusion_scheduler.training_losses_with_cfg(
                        model=model, 
                        input=input,
                        model_kwargs=batch,
                        weight_dtype=weight_dtype,
                        generator=generator
                    )

                avg_loss = accelerator.gather(loss.repeat(config.per_gpu_batch_size)).mean()
                train_loss += avg_loss.item() / config.gradient_accumulation_steps
                
                # Backpropagate
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.print(f"Step {global_step:05d}, Epoch-Step {epoch:03d}-{step:03d}, loss: {train_loss}") 
                accelerator.log({"train_loss": train_loss}, step=global_step)
                accelerator.log({"learning_rate": lr_scheduler.get_last_lr()[0]}, step=global_step)
                train_loss = 0.0
                # Save checkpoint
                if accelerator.is_main_process and \
                    global_step % config.checkpointing_steps == 0:
                    logger.info("Saving checkpoint")
                    if not os.path.exists(checkpoints_dir):
                        os.makedirs(checkpoints_dir)
                    checkpoints = os.listdir(checkpoints_dir)
                    checkpoints = [
                        d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(
                        checkpoints, key=lambda x: int(x.split("-")[1]))

                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    if len(checkpoints) >= config.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - config.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]

                        logger.info(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        )
                        logger.info(
                            f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(
                                checkpoints_dir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(
                        checkpoints_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
                
                # Run sampling
                if accelerator.is_main_process and \
                    (global_step % config.sampling_steps == 0 or \
                     global_step == 1):
                    logger.info("***** Running sampling *****")
                    pipeline = ClothDynamicsPipeline(
                        model=accelerator.unwrap_model(model),
                        scheduler=accelerator.unwrap_model(diffusion_scheduler)
                    )
                    pipeline = pipeline.to(accelerator.device)
                    pipeline.set_progress_bar_config(disable=True)

                    with torch.autocast(
                        str(accelerator.device).replace(":0", ""), 
                        enabled=accelerator.mixed_precision == "fp16"
                    ):  
                        sample_valid_loss = 0.0
                        num_sample_batches = 0
                        for step, batch in enumerate(valid_dataloader):
                            if step > 5:
                                break
                            q_prev = batch['q_prev'].to("cuda")
                            action = batch['action'].to("cuda")
                            q_mask = batch['mask'].to("cuda")
                            pred = pipeline(
                                q_prev=q_prev,
                                q_mask=q_mask,
                                action=action,
                                do_classifier_free_guidance=config.do_classifier_free_guidance
                            )[0]
                            batch_size = q_prev.shape[0]
                            if q_prev.ndim == 5:
                                q_prev = q_prev.reshape(batch_size, q_prev.shape[1], 3, -1).permute(0, 1, 3, 2).cpu().numpy()
                                q_next = batch['q_next'].reshape(batch_size, batch['action'].shape[1], 3, -1).permute(0, 1, 3, 2).cpu().numpy()
                            elif q_prev.ndim == 4:
                                q_next = batch['q_next'].cpu().numpy()
                            squared_error = (pred - q_next) ** 2
                            mse_error = np.mean(squared_error.reshape(batch_size, -1), axis=1).mean()
                            
                            sample_valid_loss += mse_error
                            num_sample_batches += 1
                    sample_valid_loss /= num_sample_batches
                    accelerator.print(f"Validation Loss (sampling): {sample_valid_loss}")
                    accelerator.log({"sampled_valid_loss": sample_valid_loss}, step=global_step)
                    del pipeline
                    torch.cuda.empty_cache()
                        
    accelerator.wait_for_everyone()
    accelerator.end_training()
if __name__ == "__main__":
    main()
