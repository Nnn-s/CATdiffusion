import einops
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch as th
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
from torch.optim.lr_scheduler import ReduceLROnPlateau
import re
import os

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config,default,ismap
# from ldm.models.diffusion.ddim import DDIMSampler
from .util import preprocess_mask,generate_max_image
from mldm.ddim import DDIMSampler
import random


class MaskUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, fusion_out=None, **kwargs):
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False,dtype=self.dtype)
        emb = self.time_embed(t_emb)
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context, fusion_out)
            hs.append(h)
        h = self.middle_block(h, emb, context, fusion_out)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context, fusion_out)
        h = h.type(x.dtype)
        return self.out(h) 
            



class MaskLDM(LatentDiffusion):
    def __init__(self, fusion_config,loss_dice_weight,*args, **kwargs):
        super().__init__(*args, **kwargs)

        self.loss_dice_weight = loss_dice_weight
        self.fusion_model = instantiate_from_config(fusion_config)
        self.new_adds = ["attn3","norm4"] #initial by inpaint
        self.new_add_params = []
        self.pre_params = []
        #train new add layers:
        for name, param in self.model.diffusion_model.named_parameters():
            if any(new_add in name for new_add in self.new_adds):
                self.new_add_params.append(param)
            else:
                self.pre_params.append(param)

        # for name, param in self.model.diffusion_model.named_parameters():
        #     if any(new_add in name for new_add in self.new_adds):
        #         param.requires_grad = True
        #         self.train_params.append(param)
        #     else:
        #         param.requires_grad = False

    def np_array2Autoencoderkl(image):
        image = rearrange(image,'b h w c -> b c h w')
        image = image.to(memory_format=torch.contiguous_format).float()
        return image
    


    @torch.no_grad()
    def get_input(self, batch, bs=None, *args, **kwargs):

        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        masked_image_224 = batch["masked_image_224"]
        clip_x,global_clip_x = self.cond_stage_model2(masked_image_224)
        
        masked_image_512 = batch["masked_image_512"]
        masked_image_input = rearrange(masked_image_512,'b h w c -> b c h w')
        masked_image_input = masked_image_input.to(memory_format=torch.contiguous_format).float()

        encoder_posterior = self.encode_first_stage(masked_image_input.to(self.dtype))
        masked_image_latents = self.get_first_stage_encoding(encoder_posterior).detach()
        
        #autoencoder need image normalize to [0,1],but clip image encoder need input to be images.,so masked_image should be PIL.Image
        mask = batch["mask_64"]


        mask_aug16 = batch['mask_aug16']
        batch_size = masked_image_224.shape[0]
        mask_aug16 = mask_aug16.long().unsqueeze(0).reshape(( batch_size,1, 16, 16))

        prompt_pool = self.cond_stage_model.pool_out(batch[self.cond_stage_key])

        if self.training:
            if random.random()>0.9:
                c = self.get_learned_conditioning([""]*c.shape[0])
                fusion_out = (torch.zeros((batch_size,8*8,1024)).to(self.device).to(self.dtype),)
            else:
                fusion_out = None
            # fusion_out = None
        else:
            fusion_out = self.fusion_model(clip_x,mask_aug16,prompt_pool,global_clip_x)

            # image_origin = batch["image_crop_224"]
            # clip_origin,global_clip_origin = self.cond_stage_model2(image_origin)
            # spatial_output = clip_origin[:,1:,:]
            # batch_size = spatial_output.shape[0]
            # avg_pool_layer = nn.AvgPool2d(kernel_size=2, stride=2)
            # spatial_output = rearrange(spatial_output,"b (h w) c -> b c h w",h=16)
            # spatial_output_avg = avg_pool_layer(spatial_output)
            # spatial_output = rearrange(spatial_output_avg,"b c h w -> b (h w) c")
            # fusion_out = (spatial_output ,)



        
        #b,1,h,w


        return x, dict(cond_prompt=[c.to(self.dtype)], cond_mask=[mask],cond_mask_img_latents = [masked_image_latents],fusion_out=[fusion_out],image_origin=batch["image_crop_224"],mask_aug16=mask_aug16,clip_x=clip_x,prompt_pool=prompt_pool,global_clip_x=global_clip_x,#uncond=dict(cond_prompt=[self.get_learned_conditioning("").repeat([c.shape[0],1,1])], cond_mask=[mask],cond_mask_img_latents = [masked_image_latents],fusion_out=[fusion_out])
                    )#mask_aug16 for stage1 loss mask

    def apply_model(self, x_noisy, t, cond,  *args, **kwargs):
        assert isinstance(cond,dict), "cond in apply_model should be a dict"
        diffusion_model = self.model.diffusion_model

        cond_prompt = torch.cat(cond['cond_prompt'],1)
        masked_img_latents = torch.cat(cond['cond_mask_img_latents'],1).to(self.dtype)
        fusion_out = cond['fusion_out'][0][0].to(self.dtype) # the last hidden state ,cond['fusion_out'] == [(last_hidden_state[:,1:-1,:], pooled_output,label_pooled)] 
        mask = torch.cat(cond['cond_mask'],1).unsqueeze(1).to(self.dtype)
        x_noisy = x_noisy.to(self.dtype)
        #mask resize and augmentation
        x_noisy = torch.cat([x_noisy,mask,masked_img_latents],1)
        eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_prompt,fusion_out = fusion_out)

        return eps

    
    def configure_optimizers(self):
        fusion_lr = self.fusion_learning_rate  # Learning rate for fusion_model and adaptor
        diffusion_lr = self.diffusion_learning_rate  # Learning rate for parameters in diffusion_model

        fusion_params = list(self.fusion_model.parameters())
        for param in fusion_params:
            assert param.requires_grad

            
        # fusion_params += self.new_add_params
        # diffusion_params = list(self.model.diffusion_model.parameters())
        
        # diffusion_params = self.pre_params
        # diffusion_new_params = self.new_add_params

        

        # opt = torch.optim.AdamW([
        #     {"params":fusion_params,"lr":fusion_lr},
        #     {"params":diffusion_params,"lr":diffusion_lr}
        #     ])
        # self.model.diffusion_model.requires_grad = False #冻结

        opt = torch.optim.AdamW([
            {"params":fusion_params,"lr":fusion_lr},
            # {"params":diffusion_params,"lr":diffusion_lr}
            ])
        scheduler = {
        'scheduler': ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=2, verbose=True),
        'monitor': 'val_loss',
        }


            


        
        return opt


    def shared_step(self, batch):
        x, c = self.get_input(batch, self.first_stage_key)
        loss = self(x, c)
        return loss

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

        




    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        return self.p_losses(x, c, t, *args, **kwargs)


    def p_losses(self, x_start, cond, t ,noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        # stage1_loss = 0
        if cond["fusion_out"][0] == None:
            clip_x = cond['clip_x']
            mask_aug16 = cond['mask_aug16']
            prompt_poll = cond['prompt_pool']
            global_clip_x = cond['global_clip_x']

            cond["fusion_out"] = [self.fusion_model(clip_x,mask_aug16,prompt_poll,global_clip_x)]

        
            ########################stage1 loss
            # clip_x = cond['clip_x']
            # mask_aug16 = cond['mask_aug16']
            # prompt_poll = cond['prompt_pool']
            # global_clip_x = cond['global_clip_x']

            # image_origin = cond["image_origin"]
            # clip_origin,global_clip_origin = self.cond_stage_model2(image_origin)
            # spatial_clip_origin = clip_origin[:,1:,:].type(self.dtype).contiguous()

            # pred_spatial,pred_global,_ = self.fusion_model(clip_x,mask_aug16,prompt_poll,global_clip_x,sample=False)
            # mask_loss =  rearrange(mask_aug16,"b c h w -> b (h w c)")
            # spatial_loss = (mask_loss*((pred_spatial - spatial_clip_origin) ** 2).mean(axis=-1)).sum() /  torch.sum(mask_loss.sum() + 1.) #mask_loss
            # global_loss = self.get_loss(global_clip_origin,pred_global,mean=False).mean()
            # stage1_loss = spatial_loss + global_loss
            ########################stage1 loss

        ####################### only_step2
        # if cond["fusion_out"] == None:
        #     image_origin = cond["image_origin"]
        #     clip_origin,global_clip_origin = self.cond_stage_model2(image_origin)
        #     spatial_output = clip_origin[:,1:,:]
        #     avg_pool_layer = nn.AvgPool2d(kernel_size=2, stride=2)
        #     spatial_output = rearrange(spatial_output,"b (h w) c -> b c h w",h=16)
        #     spatial_output_avg = avg_pool_layer(spatial_output)
        #     spatial_output = rearrange(spatial_output_avg,"b c h w -> b (h w) c")
        #     cond["fusion_out"] = [(spatial_output,)]
        ######################

        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])

        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        
        logvar_t = self.logvar[t.to(self.logvar.device)].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()# + stage1_loss * 0.1

        loss_vlb = self.get_loss(model_output, target,mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})

        # loss_dict.update({f'{prefix}/stage1_loss': stage1_loss})

        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}_loss': loss})
            

        return loss, loss_dict 
    
    def log_images(self,batch,sample=True,ddim_steps=20,ddim_eta=1,plot_denoise_rows=False):
        use_ddim=ddim_steps is not None

        log = dict()
        x,c = self.get_input(batch,self.first_stage_key)
        xc = batch[self.cond_stage_key]
        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode"):
                xc = self.cond_stage_model.decode(c)
                log["conditioning"] = xc
            if ismap(xc):
                log["original_conditioning"] = self.to_rgb(xc)
        log["reconstruction"] = self.decode_first_stage(x)
        if sample:
            with self.ema_scope("Plotting"):
                samples, z_denoise_row = self.sample_log(cond=c,batch_size=x.shape[0],ddim=use_ddim,
                                                         ddim_steps=ddim_steps,eta=ddim_eta)
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid
        

        return log
        
    
    def test_step(self,batch, batch_idx):
        cnt = 0
        for i in range(self.batch_size):
            if os.path.exists(self.root_dir +"/image" + f"/{batch_idx*self.batch_size+i}.png"):
                cnt+=1

            caption = batch["txt"][i]
            mask_filename = batch["mask_filename"][i]
            try:
                mask_name = re.search(r'([^/]+)\.png$', mask_filename).group(1)
            except:
                mask_name = re.search(r'([^/]+)\.jpg$', mask_filename).group(1)
            if os.path.exists(self.root_dir +"/image" + f"/{mask_name}.png"):
               cnt +=1 
        if cnt==self.batch_size:
            return

        x, c = self.get_input(batch, self.first_stage_key)
        H,W = (512,512)
        shape = (4,H//8,W//8)
        ddim_sampler = DDIMSampler(self)
        ddim_steps = self.ddim_steps
        num_samples = self.batch_size #if changed,cond should change like cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        #这里num_samples为batch大小，并不是产生图片数量
        uncond = dict(c)
        uncond["fusion_out"] = [(torch.zeros_like(uncond["fusion_out"][0][0]),)]
        # uncond["cond_prompt"] = [self.get_learned_conditioning(["monochrome, lowres, bad anatomy, worst quality, low quality"]).repeat(self.batch_size,1,1)]
        uncond["cond_prompt"] = [self.get_learned_conditioning([""]).repeat(self.batch_size,1,1)]

        scale = self.scale
        eta = self.eta
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                shape, c, verbose=False, eta=eta,
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uncond)
        x_samples = self.decode_first_stage(samples.to(self.dtype))
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        results = [x_samples[i] for i in range(num_samples)]
        #TODO save the label of image   data = []
        for i,image in enumerate(results):
            caption = batch["txt"][i]
            mask_filename = batch["mask_filename"][i]
            try:
                mask_name = re.search(r'([^/]+)\.png$', mask_filename).group(1)
            except:
                mask_name = re.search(r'([^/]+)\.jpg$', mask_filename).group(1)
            cv2.imwrite(self.root_dir +"/image" + f"/{mask_name}.png",cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
            with open(self.root_dir +"/text" + f"/{mask_name}.txt","w") as fw:
                fw.write(caption)
            # cv2.imwrite(f"/data/chenyifu/datasets/mldm/mask/mask{batch_idx*self.batch_size+i}.png",batch["mask_resized"][i].cpu().numpy()*255)
            # cv2.imwrite(f"/data/chenyifu/datasets/mldm/masked_image/masked_image{batch_idx*self.batch_size+i}.png",cv2.cvtColor(batch["masked_image"][i].cpu().numpy()*255,cv2.COLOR_RGB2BGR))

        return results




    
