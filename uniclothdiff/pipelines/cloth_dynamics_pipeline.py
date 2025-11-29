# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch

from diffusers.schedulers import DDPMScheduler
from diffusers.utils import BaseOutput, logging, replace_example_docstring
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from uniclothdiff.models.transformer_3d_v2 import Transformer3Dv2Model
from uniclothdiff.utils.image_utils import *

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@dataclass
class ClothDynamicsPipelineOutput(BaseOutput):
    r"""
    Output class for Stable Video Diffusion pipeline.

    Args:
        frames (`[List[List[PIL.Image.Image]]`, `np.ndarray`, `torch.FloatTensor`]):
            List of denoised PIL images of length `batch_size` or numpy array or torch tensor
            of shape `(batch_size, num_frames, height, width, num_channels)`.
    """

    frames: Union[torch.FloatTensor]
    result_tensor: Union[torch.FloatTensor]


class ClothDynamicsPipeline(DiffusionPipeline):
    
    model_cpu_offload_seq = "image_encoder->unet->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        model: Transformer3Dv2Model,
        scheduler: DDPMScheduler,
        
    ):
        super().__init__()

        self.register_modules(
            model=model,
            scheduler=scheduler,
        )
        
    @property
    def guidance_scale(self):
        return self._guidance_scale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        if isinstance(self.guidance_scale, (int, float)):
            return self.guidance_scale > 1
        return self.guidance_scale.max() > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps
    

    def denormalize_xyz(self, norm_q):
        min_vals = np.array([-1.0, -1.0, -1.0])
        max_vals = np.array([1.0, 1.0, 1.0])

        # Calculate the original coordinates from normalized points
        q = (norm_q + 1.0) * ((max_vals - min_vals) / 2.0) + min_vals
    
        return q
    
    def denormalize_delta_q(self, norm_q):
        min_vals = -0.05
        max_vals = 0.05

        # Calculate the original coordinates from normalized points
        q = (norm_q + 1.0) * ((max_vals - min_vals) / 2.0) + min_vals
    
        return q
    

    def prepare_x0(self,
        shape: tuple,
        dtype: torch.dtype,
        device: Union[str, torch.device],
        generator: torch.Generator,
    ):
        
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    # def prepare_input(self,)
    @torch.no_grad()
    def _call_discriminative(
        self,
        q_prev: torch.FloatTensor,
        q_mask: torch.Tensor,
        action: torch.FloatTensor,
    ):
        num_input_frames = q_prev.shape[1]
        q_mask = q_mask.repeat(1, num_input_frames, 1, 1)
        model_input = torch.cat([q_prev, q_mask], dim=-1)
        t = torch.zeros((q_prev.shape[0],), device=q_prev.device, dtype=torch.long)
        pred = self.model(
            model_input,
            timestep=t,
            encoder_hidden_states=action
        )[0]
        return pred
    
    @torch.no_grad()
    def __call__(
        self,
        q_prev: torch.FloatTensor,
        q_mask: torch.Tensor,
        action: torch.FloatTensor,
        num_inference_steps: int = 1000,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
        do_classifier_free_guidance: bool = False,
        guidance_scale: float = 1.0,
        eta: float = 0.0,
    ):
        batch_size = q_prev.shape[0]
        
        device = self._execution_device
        
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        num_frames = q_prev.shape[1] + action.shape[1]
        
        if q_prev.ndim == 5:
            batch_size, _, channels, _, _ = q_prev.shape
            x_0 = randn_tensor(
                    shape=(batch_size, action.shape[1], channels, *q_prev.shape[-2:]), 
                    generator=generator, 
                    device=device, 
                    dtype=q_prev.dtype
                )
            q_mask = q_mask.repeat(1, num_frames, 1, 1, 1)
        elif q_prev.ndim == 4:
            batch_size, _, _, channels = q_prev.shape
            x_0 = randn_tensor(
                    shape=(batch_size, action.shape[1], *q_prev.shape[-2:]),
                    generator=generator,
                    device=device,
                    dtype=q_prev.dtype
                )
            q_mask = q_mask.repeat(1, num_frames, 1, 1)

        x_0 = x_0 * self.scheduler.init_noise_sigma
        
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        
        if do_classifier_free_guidance:
            null_action = torch.zeros_like(action)
            action_cfg = torch.cat([action, null_action], dim=0)
            self._guidance_scale = guidance_scale

        pred_list = [x_0]
        x = x_0 # gaussian white noise
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                model_input = x
                model_input = self.scheduler.scale_model_input(model_input, t)
                
                model_input = torch.cat([q_prev, model_input], dim=1)
                
                if model_input.ndim == 5:
                    model_input = torch.cat([model_input, q_mask], dim=2)
                elif model_input.ndim == 4:
                    model_input = torch.cat([model_input, q_mask], dim=-1)
                
                if do_classifier_free_guidance:
                    model_input = torch.cat([model_input] * 2, dim=0)
                
                if model_input.ndim == 5:
                    # B, N, C, H, W ---> B, C, N, H, W
                    model_input = model_input.permute(0, 2, 1, 3, 4)
                
                # predict the noise residual
                noise_pred = self.model(
                    model_input,
                    timestep=t,
                    encoder_hidden_states=action if not do_classifier_free_guidance else action_cfg,
                )[0]

                # do classifier free guidance
                if do_classifier_free_guidance:
                    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                
                # compute the previous noisy sample x_t -> x_t-1
                x = self.scheduler.step(
                    model_output=noise_pred, 
                    timestep=t, 
                    sample=x
                ).prev_sample
                
                pred_list.append(x)
                
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()


        self.maybe_free_model_hooks()


        x = self.denormalize_delta_q(x)
        
        if action.shape[1] == 1:
            q_next = q_prev[:, -1:, ...] + x
        else:
            if q_prev.ndim == 4:
                q_init = q_prev[:, -1:, :, :]
                q_cumsum = torch.cumsum(
                    torch.cat([q_init, x], dim=1), dim=1
                )
                q_next = q_cumsum[:, 1:, :, :]
            else:
                q_init = q_prev[:, -1:, :, :, :]
                q_cumsum = torch.cumsum(torch.cat([q_init, x], dim=1), dim=1)
                q_next = q_cumsum[:, 1:, :, :, :]
        
        if q_next.ndim == 5:
            x = q_next.permute(0, 1, 3, 4, 2)
            x = x.reshape(batch_size, action.shape[1], -1, 3)
        elif q_next.ndim == 4:
            x = q_next
            
        
        final_result = x.cpu().numpy()

        if not return_dict:
            return final_result, pred_list

        return ClothDynamicsPipelineOutput(
            frames=final_result,
            result_tensor=x
        )

