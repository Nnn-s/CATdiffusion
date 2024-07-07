import json
import cv2
import numpy as np
from PIL import Image,ImageOps
from mldm.util import masking,patchify_mask,crop_512
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import random
import numpy as np
import os


"""
{'mask_image': 'test/labels/masks/6/611cf47482fe991f_m01bjv_8ae41a6b.png', 
'label': 'Bus', 
'box_id': '8ae41a6b', 
'area': 0.42109375000000004, 
'box': [0.234375, 0.285417, 0.765625, 0.55], 
'image_id': '611cf47482fe991f', 
'image': 'test/data/611cf47482fe991f.jpg'}
"""

class OpenimagesDataset(Dataset):
    def __init__(self,mode = None):
        self.data = []
        assert mode in ['train','test','validation']
        if mode == 'train':
            filename = 'openimages_train.txt'
            data_txt = "laion2B.txt"
        elif mode == 'validation':
            filename = 'openimages_validation.txt'
        else:
            filename ='openimages_validation.txt'
        self.mode = mode

        self.data = []
        self.data2 = []
        data = open(filename).readlines()
        for line in data:
            item = eval(line.strip())
            self.data.append(item)

        if mode == "train":
            data2 = open(data_txt).readlines()
            for line in data2:
                item = eval(line.strip())
                self.data2.append(item)

        
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.mode == "test" or self.mode == "validation": 
            item = self.data[idx]

            mask_filename = item['mask_image']

            label = item['label']
            image_filename = item['image']
            box = item["box"]
            box_id = item["box_id"]
            image_id = item["image_id"]

            image = Image.open(image_filename).convert("RGB")
            mask = Image.open(mask_filename).convert("L")

            image_crop_512 = image.resize((512,512))
            mask_crop_512 = mask.resize((512,512))


            masked_image_512 = masking(image_crop_512,mask_crop_512,return_pil=False)
            masked_image_512 = masked_image_512/127.5 -1.



            image_crop_224 = image_crop_512.resize((224,224))
            mask_crop_224 = mask_crop_512.resize((224,224))

            masked_image_224 = masking(image_crop_224,mask_crop_224)

            mask_16 = patchify_mask(np.array(mask_crop_224))


            mask_64 = np.array(mask_crop_512.resize((64,64)))/255.0

            mask_64[mask_64 > 0.5] = 1.
            mask_64[mask_64 < 0.5] = 0.

            image_crop_512 = np.array(image_crop_512)/127.5 - 1.0
            masked_image_224 = np.array(masked_image_224)
            image_crop_224=np.array(image_crop_224)                     #clip processor will rescale if do_rescale, and do_rescale default to true. type must to be np.unit8

            return  dict(jpg=image_crop_512, mask_64=mask_64,txt=label,image_crop_224=image_crop_224,masked_image_512=masked_image_512,masked_image_224=masked_image_224,mask_aug16 = mask_16,mask_filename = mask_filename) 

        else:
            if random.random() > 0.2:  #Train with OpenImages v6
                item = self.data[idx]

                mask_filename = item['mask_image']

                label = item['label']
                image_filename = item['image']
                box = item["box"]
                box_id = item["box_id"]
                image_id = item["image_id"]

                image = Image.open(image_filename).convert("RGB")
                mask = Image.open(mask_filename).convert("L")

                if random.random() > 0.5:
                    kernel = np.ones((5, 5), np.uint8)
                    mask = cv2.dilate(np.array(mask), kernel, iterations=1)
                    mask = Image.fromarray(mask.astype(np.uint8))
                else:
                    wt,ht = mask.size
                    box_mask = np.zeros((ht, wt), dtype=np.uint8) 
                    x,y,w,h = box
                    x2 ,y2= x+w,y+h
                    box_mask[int(y*ht):int(y2*ht),int(wt*x):int(wt*x2)] = 255
                    mask = Image.fromarray(box_mask)


        
                resize_op_512 = transforms.Resize(512)
                
                image_resize_512 = resize_op_512(image)
                mask_resize_512 = resize_op_512(mask)
                image_resize_512 = image_resize_512.resize(mask_resize_512.size) 


                image_crop_512,mask_crop_512 = crop_512(image_resize_512,mask_resize_512)

                masked_image_512 = masking(image_crop_512,mask_crop_512,return_pil=False)#numpy
                masked_image_512 = masked_image_512/127.5 -1.



                image_crop_224 = image_crop_512.resize((224,224))
                mask_crop_224 = mask_crop_512.resize((224,224))

                masked_image_224 = masking(image_crop_224,mask_crop_224)

                mask_16 = patchify_mask(np.array(mask_crop_224))


                mask_64 = np.array(mask_crop_512.resize((64,64)))/255.0

                mask_64[mask_64 > 0.] = 1.
                if random.random() < 0.2: #Long prompt training
                    caption_dir = os.path.join("fiftyone/blip2-opt-2.7b_box_caption",self.mode) 
                    try:
                        caption_path = os.path.join(caption_dir,f"{image_id}_{box_id}.txt")
                        with open(caption_path,"r") as fr:
                            label = fr.read()
                    except:
                        print("Missing long prompt")
                    

                image_crop_512 = np.array(image_crop_512)/127.5 - 1.0
                masked_image_224 = np.array(masked_image_224)
                image_crop_224=np.array(image_crop_224)                     #clip processor will rescale if do_rescale, and do_rescale default to true. type mush to be np.unit8

            else: #Train with LAION
                item = self.data2[idx % len(self.data2)]
                image_filename = item["image_name"]

                image = Image.open(image_filename).convert("RGB")
                label = item["text"]

                width,height = image.size
                mask = Image.new("L", (width, height), 0)

                resize_op_512 = transforms.Resize(512)
                
                image_resize_512 = resize_op_512(image)
                mask_resize_512 = resize_op_512(mask)
                image_resize_512 = image_resize_512.resize(mask_resize_512.size) # image 1024,683 mask1500 1000 


                image_crop_512,mask_crop_512 = crop_512(image_resize_512,mask_resize_512)

                masked_image_512 = masking(image_crop_512,mask_crop_512,return_pil=False)#numpy
                masked_image_512 = masked_image_512/127.5 -1.



                image_crop_224 = image_crop_512.resize((224,224))
                mask_crop_224 = mask_crop_512.resize((224,224))

                masked_image_224 = masking(image_crop_224,mask_crop_224)

                mask_16 = patchify_mask(np.array(mask_crop_224))


                mask_64 = np.array(mask_crop_512.resize((64,64)))/255.0

                image_crop_512 = np.array(image_crop_512)/127.5 - 1.0
                masked_image_224 = np.array(masked_image_224)
                image_crop_224=np.array(image_crop_224) 

                




                   
            return dict(jpg=image_crop_512, mask_64=mask_64,txt=label,image_crop_224=image_crop_224,masked_image_512=masked_image_512,masked_image_224=masked_image_224,mask_aug16 = mask_16) 



class OpenimagesBoxDataset(Dataset):
    def __init__(self,mode = None):
        self.data = []
        assert mode in ['train','test','validation']
        if mode == 'train':
            filename = '/data/chenyifu/datasets/fiftyone/openimages_train.txt'#'/data/chenyifu/datasets/fiftyone/test_train.txt'
        elif mode == 'validation':
            filename = '/data/chenyifu/datasets/fiftyone/openimages_validation.txt'#'/data/chenyifu/datasets/fiftyone/test_val.txt'
        else:
            filename = '/data/chenyifu/datasets/fiftyone/openimages_validation.txt'#"./diversity/diversity_box.txt"#'/data/chenyifu/datasets/fiftyone/test_test.txt'
        self.mode = mode

        self.data = []
        data = open(filename).readlines()
        for line in data:
            item = eval(line.strip())
            self.data.append(item)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.mode == "test" or self.mode == "validation": 
            item = self.data[idx]

            mask_filename = item['mask_image']

            label = item['label']
            image_filename = item['image']
            box = item["box"]
            box_id = item["box_id"]
            image_id = item["image_id"]

            image = Image.open(image_filename).convert("RGB")
            #mask = Image.open(mask_filename).convert("L")

            box_mask = np.zeros((512, 512), dtype=np.uint8) 
            x,y,w,h = box
            x2 ,y2= x+w,y+h
            box_mask = cv2.rectangle(box_mask,(int(x*512),int(y*512)) ,(int(x2*512),int(y2*512)), 255, -1)
            mask_crop_512 = Image.fromarray(box_mask).convert("L")

            image_crop_512 = image.resize((512,512))
            # mask_crop_512 = mask.resize((512,512))


            masked_image_512 = masking(image_crop_512,mask_crop_512,return_pil=False)#numpy
            masked_image_512 = masked_image_512/127.5 -1.



            image_crop_224 = image_crop_512.resize((224,224))
            mask_crop_224 = mask_crop_512.resize((224,224))

            masked_image_224 = masking(image_crop_224,mask_crop_224)

            mask_16 = patchify_mask(np.array(mask_crop_224))


            mask_64 = np.array(mask_crop_512.resize((64,64)))/255.0

            mask_64[mask_64 > 0.5] = 1.
            mask_64[mask_64 < 0.5] = 0.

            image_crop_512 = np.array(image_crop_512)/127.5 - 1.0
            masked_image_224 = np.array(masked_image_224)
            image_crop_224=np.array(image_crop_224)                     #clip processor will rescale if do_rescale, and do_rescale default to true. type must to be np.unit8


            return  dict(jpg=image_crop_512, mask_64=mask_64,txt=label,image_crop_224=image_crop_224,masked_image_512=masked_image_512,masked_image_224=masked_image_224,mask_aug16 = mask_16,mask_filename = mask_filename) 
