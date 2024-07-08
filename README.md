# CATdiffusion
Official implementation of Improving Text-guided Object Inpainting with Semantic Pre-inpainting in ECCV 2024.

> **Improving Text-guided Object Inpainting with semantic Pre-inpainting**

 Our model checkpoints trained on [OpenImage v6](https://storage.googleapis.com/openimages/web/download_v6.html) and subset of [LAION 5B](https://laion.ai/blog/laion-5b/) have been released.
 
 🤗 [Try out CATdiffusion](https://huggingface.co/Nnnswic/CaTdiffusion)

 ![framework](assets/paper_images/framework.png)&nbsp;

## Installation
1. Clone the repository
```sh
https://github.com/Nnn-s/CATdiffusion
```
2. Create a conda environmenta and install required packages
```sh
conda env create -f environment.yaml
conda activate catdiffusion
```

## Inference
**Infer with dataset:**
```sh
python dataset_infer.py --ckpt test.ckpt --output_dir results --config ./models/mldm_v15.yaml
```

```sh
{'mask_image': 'test/labels/masks/6/611cf47482fe991f_m01bjv_8ae41a6b.png', 
'label': 'Bus', 
'box_id': '8ae41a6b', 
'area': 0.42109375000000004, 
'box': [0.234375, 0.285417, 0.765625, 0.55], 
'image_id': '611cf47482fe991f', 
'image': 'test/data/611cf47482fe991f.jpg'}
```

## Train:

The training process is divided into two stages. In the first stage, the Semantic Inpainter is trained separately using distillation learning with a CLIP teacher model, and the UNet and Reference Adapter are trained separately using ground truth images. 

In the second stage, the two checkpoints from the first stage are combined, the UNet and adapter are frozen, and the Semantic Inpainter is trained using diffusion loss. Finally, all parameters are unfrozen and the entire model is trained using diffusion loss.

### Stage1:

**1. First, the original [Stable Diffusion 1.5 inpainting model](https://huggingface.co/runwayml/stable-diffusion-inpainting/tree/main) needs to be processed.**

```sh
python tool_stage1.py --input_path sd-v1-5-inpainting.ckpt --output_path ckpt_for_stage1.ckpt --config ./models/mldm_v15.yaml
```

**2. Train Semantic Inpainter**

```sh
python train.py --ckpt ckpt_for_stage1.ckpt --config ./models/mldm_v15_stage1.yaml --save_path ./stage1_Semantic_Inpainter
```

**3. Train Unet and Reference Adapter**

```sh
python train.py --ckpt ckpt_for_stage1.ckpt --config ./models/mldm_v15_unet_only.yaml --save_path ./stage1_Unet
```

### Stage2:

**1. Combined ckpts in Stage1:**

```sh
python tool_merge_for_stage2.py --stage1_path ./stage1_Semantic_Inpainter/last.ckpt --input_path ./stage1_Unet/last.ckpt --output_path ckpt_for_stage2.ckpt --config ./models/mldm_v15.yaml
```

**2. Train Semantic Inpainter with diffusion loss:**

```sh
python train.py --ckpt ckpt_for_stage2.ckpt --config ./models/mldm_v15_stage2_1.yaml --save_path ./stage2_1
```

**3. Final Training:**

```sh
python train.py --ckpt stage2_1/last.ckpt --config ./models/mldm_v15_stage2_1.yaml --save_path ./stage2_2
```

## Citation
```
@article{Chen2024CATiffusion,
  title={Improving Text-guided Object Inpainting with semantic Pre-inpainting},
  author={},
  journal={},
  year={2024}
}
```



## Generated Examples
### Bounding Box mask inpainted results:
 ![box_image](assets/readme_images/long_image_0.png)&nbsp;
 ![box_image](assets/readme_images/long_image_1.png)&nbsp;
 ![box_image](assets/readme_images/long_image_2.png)&nbsp;

### Segmentation mask inpainted results:
 ![seg_image](assets/readme_images/long_image_3.png)&nbsp;
 ![seg_image](assets/readme_images/long_image_4.png)&nbsp;
