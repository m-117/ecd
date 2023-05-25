from scipy.interpolate import interp1d

import torch

import pickle
import random

def interpolate_tensors(tensor1, tensor2):
    interp_func = interp1d([0, 1], torch.cat([tensor1.unsqueeze(0), tensor2.unsqueeze(0)]), axis=0)
    new_tensor_sub = interp_func(torch.linspace(0, 1, 10))
    return new_tensor_sub[1:10]

def interpolate_tensor(input_tensor):
    num_rows = input_tensor.shape[0]
    output_tensor = torch.empty(num_rows*4, 6)
    for i in range(num_rows-1):
        interp_func = interp1d([0, 1], torch.stack([input_tensor[i], input_tensor[i+1]]), axis=0)
        intermediate_values = interp_func(torch.linspace(0, 1, 4)[1:-1])
        output_tensor[i*4:(i+1)*4] = torch.cat([torch.tensor(input_tensor[i]).unsqueeze(0), torch.tensor(intermediate_values), torch.tensor(input_tensor[i+1]).unsqueeze(0)], dim=0)
    return output_tensor

def edit_pose_tensor(pose_tensor, emotion, head_movement_file):
    # Load head movement dictionary
    with open(head_movement_file, 'rb') as f:
        head_movements = pickle.load(f)

    # Randomly select actor
    actor = random.choice(list(head_movements[emotion].keys()))

    # Choose random head movements
    selected_head_movements = []
    accumulated_length = 0
    while accumulated_length*4 < pose_tensor.shape[0]:
        random_head_movement = random.choice(head_movements[emotion][actor])
        selected_head_movements.append(random_head_movement)
        accumulated_length += random_head_movement.shape[0]

    new_pose_tensor = torch.zeros_like(pose_tensor)

    # Interpolate between pose_tensor and head movements
    new_pose_tensor = torch.zeros_like(pose_tensor)
    accumulated_length = 0
    for i in range(len(selected_head_movements)):
        head_movement = torch.tensor(selected_head_movements[i])
        hm = interpolate_tensor(head_movement)
        start_index = 1 + accumulated_length
        accumulated_length += hm.shape[0]
        end_index = min((1 + 8 + accumulated_length), pose_tensor.shape[0])

        # Interpolate pose subtensor
        new_pose_subtensor = interpolate_tensors(new_pose_tensor[start_index-1], hm[0])

        # Add interpolation and head movement
        new_pose_tensor[start_index:end_index] = torch.cat([torch.tensor(new_pose_subtensor), torch.tensor(hm)])[:end_index-start_index]

    # Interpolate the last 5 pose elements
    new_pose_subtensor = interpolate_tensors(new_pose_tensor[-9], torch.zeros(6))

    new_pose_tensor[-9:] = torch.tensor(new_pose_subtensor)

    return new_pose_tensor, actor
