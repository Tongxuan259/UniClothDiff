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
        # print(sample_input.shape)
        # print(q_mask.shape)
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
        # last_frame = q_prev[:, -1, :, :, :].reshape(batch_size, 3, 2500).permute(0, 2, 1)
        # input = input.reshape(1, 3, 2500).permute(0, 2, 1)
        # input = last_frame
        if noise is None:
            noise = torch.randn_like(input).to(weight_dtype).to(input.device)

        noisy_input = self.add_noise(input, noise, timesteps=t).to(weight_dtype).to(input.device)
        # noisy_input = noisy_input.reshape(1,1,3,100,25)

        # # Should be moved into the model forward
        # # Conditioning dropout to support classifier-free guidance during inference.
        # condition_dropout_pro = 0.0
        # # if args.conditioning_dropout_prob is not None:
        # random_p = torch.rand(batch_size, device=input.device, generator=generator)
        # # Sample masks for the dropout action.
        # action_mask = random_p < 2 * condition_dropout_pro
        # action_mask = action_mask.reshape(batch_size, 1, 1)
        # # # Final action condition
        # # null_action_condition = torch.zeros_like(action) - 1.0
        # action = action_mask * action
        
        # original implementation
        sample_input = torch.cat([q_prev, noisy_input], dim=1)
        
        if sample_input.ndim == 5:
            _, num_frames, _, _, _ = sample_input.shape
            q_mask = q_mask.repeat(1, num_frames, 1, 1, 1)
            sample_input = torch.cat([sample_input, q_mask], dim=2)
            sample_input = sample_input.permute(0, 2, 1, 3, 4) # reshape for latte
        elif sample_input.ndim == 4:
            _, num_frames, _, _ = sample_input.shape
            q_mask = q_mask.repeat(1, num_frames, 1, 1)
            sample_input = torch.cat([sample_input, q_mask], dim=-1)
        
        model_output = model(
            hidden_states=sample_input,
            timestep=t,
            encoder_hidden_states=action,
        ).sample
        
        
        
        
        ## for debug ptv3
        # q_mask = q_mask.reshape(1, 1, 2500).permute(0, 2, 1)
        # sample_input = torch.cat([last_frame, noisy_input, q_mask], dim=-1)
        # sample_input = torch.cat([sample_input, 
        #                           0.001 * t.expand((sample_input.shape[0], sample_input.shape[1], 1))],
        #                          dim=-1)
        
        # model_output = model(
        #     sample_input
        # )


        
        # model_output = model_output - input
        # model_output = model_output.reshape(1,3,2500)
        loss = torch.mean(((model_output - noise)**2).reshape(model_output.shape[0], -1), dim=1)
        
        # loss = torch.mean(((model_output - input)**2).reshape(model_output.shape[0], -1), dim=1)

        loss = loss.mean()
        
        return loss
                

