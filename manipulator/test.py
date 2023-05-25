from manipulator import manipulate_expressions
import torch
import os

import numpy as np

import csv

def manipulate_csv_dataset(deca_path, csv_path, output_path):
    """Transforms preprocessed input deca sequences and corresponding input text from csv. Saves manipulated params as tensor files."""
    # open csv file
    with open(csv_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t')
        text_list = []
        for row in csvreader:
            text_list.append((row[2], row[0]))

    # iterate input text
    for item in text_list:
        if os.path.isdir(os.path.join(output_path, item[0])):
            continue
        # load and reshape tensors
        deca_tensor = torch.load(os.path.join(deca_path, item[0] + ".pt"))
        deca_tensor = deca_tensor.reshape(deca_tensor.shape[1]//51, 51)
        deca_tensor = deca_tensor.cpu()
        # transform params
        new_exp, new_pose, actor_list, gt_emotion_list = manipulate_expressions(deca_tensor, item[1], 1.025)
        os.mkdir(os.path.join(output_path, item[0]))
        # save tensors, actor list and gt emotions
        torch.save(new_exp, os.path.join(output_path, item[0], "exp.pt"))
        torch.save(new_pose, os.path.join(output_path, item[0], "pose.pt"))
        with open(os.path.join(output_path, item[0], "output.txt"), "w") as f:
            f.write("Input Text: " + item[1] + "\n")
            f.write("Chosen Actor: " + str(actor_list) + "\n")
        with open(os.path.join(output_path, item[0], "label"), "w") as f:
            f.write(gt_emotion_list)



def manipulate_single_sequence(deca_path, input_text):
    """Takes a single sequence and creates manipulated sequences for every emotion."""

    # set basic emotions
    emotions = ["angry", "disgusted", "fear", "happy", "sad", "surprised", "neutral"]
    # load deca params
    exp = np.load(os.path.join(deca_path, "exp.npy"))
    pose = np.load(os.path.join(deca_path, "jaw_3d.py.npy"))
    # combine the two arrays
    combined_array = np.concatenate((pose[:, 0][:, np.newaxis], exp[:, :50]), axis=1)
    deca_tensor = torch.from_numpy(combined_array).cpu()
    mult = 1
    # create sequence set
    for i in range(20):
        os.mkdir(os.path.join(deca_path, str(i+1)))
        for emotion in emotions:
            print(emotion)
            new_exp, new_pose, actor_list, emo_list = manipulate_expressions(deca_tensor, input_text, mult, emotion)
            os.mkdir(os.path.join(deca_path, str(i+1), emotion))    
            torch.save(new_exp, os.path.join(deca_path, str(i+1), emotion, "exp.pt"))
            torch.save(new_pose, os.path.join(deca_path, str(i+1), emotion, "pose.pt"))
        mult += 0.01

