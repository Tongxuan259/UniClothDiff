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

from uniclothdiff.models.transformer_state_est_v3 import TransformerStateEstV3Model
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


class ClothStateEstPipeline(DiffusionPipeline):
    
    model_cpu_offload_seq = "image_encoder->unet->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        model: TransformerStateEstV3Model,
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
    
    def _call_v2(
        self,
        encoder_hidden_states: torch.FloatTensor,
        q_temp: torch.FloatTensor,
        shape: tuple,
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
        device = self._execution_device

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        x_0 = randn_tensor(
            shape=shape, 
            generator=generator, 
            device=device, 
            dtype=encoder_hidden_states.dtype
        )
        
        x_0 = x_0 * self.scheduler.init_noise_sigma
         
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        

        pred_list = [x_0]
        x = x_0 # gaussian white noise
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                model_input = torch.cat([x, q_temp], dim=-1)
                model_input = self.scheduler.scale_model_input(model_input, t)
                
                if do_classifier_free_guidance:
                    model_input = torch.cat([model_input] * 2, dim=0)
                    
                # predict the noise residual
                noise_pred = self.model(
                    model_input,
                    timestep=t,
                    encoder_hidden_states=encoder_hidden_states
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

        final_result = x.cpu().numpy()

        if not return_dict:
            return final_result, pred_list

        return ClothDynamicsPipelineOutput(
            frames=final_result,
            result_tensor=x
        )
        

    @torch.no_grad()
    def __call__(
        self,
        encoder_hidden_states: torch.FloatTensor,
        q_temp: torch.FloatTensor,
        shape: tuple,
        num_inference_steps: int = 1000,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
        do_classifier_free_guidance: bool = False,
        guidance_scale: float = 1.0,
        eta: float = 0.0,
        call_v2: bool = False,
        
    ):
        if call_v2:
            return self._call_v2(
                encoder_hidden_states=encoder_hidden_states,
                q_temp=q_temp,
                shape=shape,
                num_inference_steps=num_inference_steps,
                generator=generator,
                latents=latents,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                return_dict=return_dict,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guidance_scale=guidance_scale,
                eta=eta,
            )
        
        device = self._execution_device

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        x_0 = randn_tensor(
                shape=shape, 
                generator=generator, 
                device=device, 
                dtype=encoder_hidden_states.dtype
            )
        
        x_0 = x_0 * self.scheduler.init_noise_sigma
         
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(2, dim=0)
        if do_classifier_free_guidance:
            null_encoder_hidden_states = torch.zeros_like(encoder_hidden_states)
            encoder_hidden_states_cfg = torch.cat([encoder_hidden_states, null_encoder_hidden_states], dim=0)
            self._guidance_scale = guidance_scale

        pred_list = [x_0]
        x = x_0 # gaussian white noise
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                model_input = torch.cat([q_temp.unsqueeze(1), x.unsqueeze(1)], dim=1)
                model_input = self.scheduler.scale_model_input(model_input, t)
                
                if do_classifier_free_guidance:
                    model_input = torch.cat([model_input] * 2, dim=0)
                    
                # predict the noise residual
                noise_pred = self.model(
                    model_input,
                    timestep=t,
                    encoder_hidden_states=encoder_hidden_states \
                        if not do_classifier_free_guidance \
                            else encoder_hidden_states_cfg,
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

        final_result = x.cpu().numpy()

        if not return_dict:
            return final_result, pred_list

        return ClothDynamicsPipelineOutput(
            frames=final_result,
            result_tensor=x
        )

