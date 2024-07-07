import os
import argparse
import torch
from mldm.model import create_model, load_state_dict

def main(args):
    stage1_path = args.stage1_path
    input_path = args.input_path
    output_path = args.output_path

    assert os.path.exists(input_path), 'Input model does not exist.'
    assert not os.path.exists(output_path), 'Output filename already exists.'
    assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'

    model = create_model(config_path=args.config)
    model.load_state_dict(load_state_dict(stage1_path, location='cpu'))

    pretrained_weights = load_state_dict(input_path)
    if 'state_dict' in pretrained_weights:
        pretrained_weights = pretrained_weights['state_dict']

    scratch_dict = model.state_dict()

    target_dict = {}
    for k in scratch_dict.keys():
        if k in pretrained_weights and pretrained_weights[k].shape == scratch_dict[k].shape:
            target_dict[k] = pretrained_weights[k].clone()
        else:
            target_dict[k] = scratch_dict[k].clone()
            print(f'These weights are newly added: {k}')

    model.load_state_dict(target_dict, strict=True)
    torch.save(model.state_dict(), output_path)
    print('Done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Checkpoint Conversion Script")
    parser.add_argument('--stage1_path', type=str, required=True, help='Path to the stage1 checkpoint file')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input checkpoint file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output checkpoint file')
    parser.add_argument('--config', type=str, required=True, help='Path to the model config file')
    
    args = parser.parse_args()
    main(args)
