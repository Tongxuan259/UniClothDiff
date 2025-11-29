import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from uniclothdiff.registry import SCHEDULERS
import torch.nn as nn

def get_contributing_params(y, top_level=True):
    nf = y.grad_fn.next_functions if top_level else y.next_functions
    for f, _ in nf:
        try:
            yield f.variable
        except AttributeError:
            pass  # node has no tensor
        if f is not None:
            yield from get_contributing_params(f, top_level=False)

@SCHEDULERS.register_module()
class DDPM(DDPMScheduler):
    def training_losses(self, model, input, model_kwargs=None, noise=None, weight_dtype=None):
        # sample random timestep t
        batch_size = input.shape[0]
        t = torch.randint(0, self.config.num_train_timesteps, 
                                  (batch_size,), device=input.device)
        if noise is None:
            noise = torch.randn_like(input).to(weight_dtype).to(input.device)
        noisy_input = self.add_noise(input, noise, timesteps=t).to(weight_dtype).to(input.device)
        
        # should be moved into the model forward
        if model_kwargs is not None:
            q_prev = model_kwargs.pop('q_prev')
            q_mask = model_kwargs.pop('mask')
            action = model_kwargs.pop('action')
            
        sample_input = torch.cat([q_prev, noisy_input], dim=1)
        _, num_frames, _, _, _ = sample_input.shape
        q_mask = q_mask.repeat(1, num_frames, 1, 1, 1)

        sample_input = torch.cat([sample_input, q_mask], dim=2)

        action = action

        sample_input = sample_input.permute(0, 2, 1, 3, 4) # reshape for latte
        
        model_output = model(
            hidden_states=sample_input,
            timestep=t,
            encoder_hidden_states=action
        ).sample
        
        
        
        loss = torch.mean(((model_output - noise)**2).reshape(model_output.shape[0], -1), dim=1)
        loss = loss.mean()
        
        return loss
    
    def point2dict(self, points, grid_size=0.01):
        B, N, C = points.shape
        offset = []
        for i in range(B):
            offset += [(i+1) * N]
        offset = torch.tensor(offset).int().to(points.device)
        points = points.view(-1, C)
        data_dict = {
            "feat": points,
            "coord": points[:, :3],
            "grid_size": grid_size,
            "offset": offset
        }
        return data_dict
    
    def training_losses_with_cfg(self, model, input, model_kwargs=None, noise=None, weight_dtype=None, generator=None):
        # sample random timestep t
        batch_size = input.shape[0]
        t = torch.randint(0, self.config.num_train_timesteps, 
                                  (batch_size,), device=input.device)
        if model_kwargs is not None:
            q_prev = model_kwargs.pop('q_prev')
            q_mask = model_kwargs.pop('mask')
            action = model_kwargs.pop('action')
       
        if noise is None:
            noise = torch.randn_like(input).to(weight_dtype).to(input.device)

        noisy_input = self.add_noise(input, noise, timesteps=t).to(weight_dtype).to(input.device)
        
        # original implementation
        sample_input = torch.cat([q_prev, noisy_input], dim=1)
        
        if sample_input.ndim == 5:
            _, num_frames, _, _, _ = sample_input.shape
            q_mask = q_mask.repeat(1, num_frames, 1, 1, 1)
            sample_input = torch.cat([sample_input, q_mask], dim=2)
            sample_input = sample_input.permute(0, 2, 1, 3, 4) 
        elif sample_input.ndim == 4:
            _, num_frames, _, _ = sample_input.shape
            q_mask = q_mask.repeat(1, num_frames, 1, 1)
            sample_input = torch.cat([sample_input, q_mask], dim=-1)
        
        model_output = model(
            hidden_states=sample_input,
            timestep=t,
            encoder_hidden_states=action,
        ).sample
        

        loss = torch.mean(((model_output - noise)**2).reshape(model_output.shape[0], -1), dim=1)
        

        loss = loss.mean()
        
        return loss
                

