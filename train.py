import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import OpenimagesDataset
from mldm.logger import ImageLogger
from mldm.model import create_model, load_state_dict
import argparse
# import torch
# torch.cuda.init()

def main(args):
    # Configs
    resume_path = args.ckpt
    save_dir = args.save_path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    batch_size = 16
    logger_freq = 400

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(args.config).cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.fusion_learning_rate = 1e-4
    model.diffusion_learning_rate = 1e-5

    # Misc
    train_dataset = OpenimagesDataset(mode='train')
    val_dataset = OpenimagesDataset(mode='validation')
    train_dataloader = DataLoader(train_dataset, num_workers=1, batch_size=batch_size, shuffle=True, prefetch_factor=2)
    val_dataloader = DataLoader(val_dataset, num_workers=1, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=save_dir,
        filename='{epoch:02d}--{val_loss:6f}',
        save_top_k=2,
        monitor='val_loss',
        mode='min',
        save_last=True
    )

    trainer = pl.Trainer(
        gpus=8,
        precision=32,
        max_epochs=2,
        val_check_interval=0.5,
        callbacks=[checkpoint_callback]
    )

    # Train!
    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--config', type=str, required=True, help='Path to the model config file')
    parser.add_argument('--save_path', type=str, required=True, help='Directory to save the results')
    
    args = parser.parse_args()
    main(args)
