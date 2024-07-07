import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import OpenimagesBoxDataset, OpenimagesDataset
# from mldm.logger import ImageLogger
from mldm.model import create_model, load_state_dict
import argparse
from pytorch_lightning import seed_everything
import torch

def main(args):
    # Configs
    resume_path = args.ckpt
    batch_size = 25
    logger_freq = 300
    eta = 0.0
    scale = 7.5
    ddim_steps = 50
    seed = 0
    seed_everything(seed)

    root_dir = args.output_dir
    for subdir in ["image", "text"]:
        if not os.path.exists(os.path.join(root_dir, subdir)):
            os.makedirs(os.path.join(root_dir, subdir))

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(args.config).cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.eta = eta
    model.scale = scale
    model.ddim_steps = ddim_steps
    model.batch_size = batch_size
    model.root_dir = root_dir

    # dtype = torch.float16
    # if dtype == torch.float16:
    #     model = model.half()
    #     model.fusion_model.dtype = model.dtype
    #     model.model.diffusion_model.dtype = model.dtype

    # Misc
    test_dataset = OpenimagesBoxDataset(mode='test')
    test_dataloader = DataLoader(test_dataset, num_workers=0, batch_size=batch_size, shuffle=False, drop_last=True)
    # logger = ImageLogger(batch_frequency=logger_freq)

    trainer = pl.Trainer(gpus=1)

    # Infer!
    trainer.test(model, test_dataloader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model Testing Script")
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the results')
    parser.add_argument('--config', type=str, required=True, help='Path to the model config file')
    
    args = parser.parse_args()
    main(args)
