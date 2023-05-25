import os, sys
import cv2
from skimage.transform import warp
from skimage import img_as_ubyte
import numpy as np
import argparse
from tqdm import tqdm
import torch
import pickle
import torch.nn.functional as F
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from DECA.decalib.deca import DECA
from DECA.decalib.utils import util
from DECA.decalib.utils.config import cfg as deca_cfg

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_DECA_params(exp_tensor, pose_tensor, device = 'cuda'):
    params = []
    for i in range(exp_tensor.shape[0]):
        codedict = {}
        exp_params = torch.zeros(1, 50).to(device)
        exp_params[0, :] = exp_tensor[i, :]
        codedict['exp'] = exp_params
        pose_params = torch.zeros(1, 6).to(device)
        pose_params[0, :] = pose_tensor[i, :]
        codedict['pose'] = pose_params
        #codedict['shape'] = torch.zeros(1, 100).to(device)
        codedict['shape'] = torch.tensor([[-0.5093,  0.1352, -0.5750, -0.0544, -0.4168, -0.0882,  0.6027,  0.0917,
         -0.2597,  0.0348, -0.0049,  0.0420, -0.1540, -0.1410,  0.3470, -0.1216,
          0.3932,  0.1611,  0.4189, -0.6555,  0.1839,  0.0265, -0.1091,  0.0521,
         -0.4047, -0.1786,  0.0047,  0.3341,  0.0841, -0.5071,  0.1547,  0.0474,
          0.2105,  0.0781,  0.0138,  0.1390, -0.1483,  0.3826, -0.2810,  0.0168,
         -0.7803,  0.3548,  0.3140,  0.0515, -0.0715, -0.3263, -0.3828,  0.3817,
         -0.4388, -0.0847,  0.3015, -0.4247,  0.1335,  0.0857,  0.1925,  0.1195,
          0.1909, -0.3589, -0.4047,  0.1183, -0.0296,  0.1501, -0.0837, -0.3012,
          0.0366, -0.1905,  0.1157, -0.0201, -0.1092,  0.1899, -0.1686, -0.0140,
          0.2485,  0.0333, -0.0594,  0.0801, -0.1662, -0.0373,  0.0484, -0.0614,
          0.0409,  0.0344, -0.0376,  0.1200,  0.0948,  0.0590,  0.0497, -0.0944,
          0.1473, -0.1486,  0.1914, -0.0496,  0.1461, -0.2367, -0.1945, -0.1059,
          0.1849, -0.0531, -0.0434, -0.0511]]).to(device)
        codedict['tex'] = torch.zeros(1, 50).to(device)
        codedict['cam'] = torch.tensor([[1.0127e+01, 3.5268e-03, 1.7923e-02]]).to(device)
        codedict['light'] = torch.tensor([[[ 3.5152,  3.5004,  3.5005],
                                            [ 0.0700,  0.0585,  0.0720],
                                            [ 0.3212,  0.3108,  0.3108],
                                            [-0.3543, -0.4179, -0.4150],
                                            [-0.0891, -0.0894, -0.0894],
                                            [-0.1121, -0.1189, -0.1125],
                                            [-0.2273, -0.2351, -0.2365],
                                            [ 0.5954,  0.5977,  0.5974],
                                            [ 0.2470,  0.2164,  0.2201]]], dtype=torch.float32).to(device)
        codedict['detail'] = torch.tensor([[ 4.4144e-01,  3.9316e-02, -1.7310e-01,  1.0139e-01,  7.7922e-03,
          6.2468e-02,  4.6178e-02,  2.3371e-02,  8.0436e-03,  1.1923e-01,
         -4.3919e-02,  1.9072e-02, -4.0900e-03, -9.2658e-02, -2.2372e-01,
         -4.9354e-02,  3.8744e-02, -3.7090e-02, -1.4366e-01,  3.3426e-02,
          4.2252e-02,  1.0437e-02,  6.4664e-02, -4.2638e-02,  2.4910e-02,
          1.7744e-01,  5.5290e-02,  1.3676e-02,  1.3796e-03,  2.3106e-02,
          2.8728e-01,  3.9226e-03, -4.5283e-01,  1.6838e-02, -6.1141e-02,
          2.6064e-02,  1.9707e-01,  7.7336e-04, -1.2178e-02,  8.2514e-02,
         -3.5814e-02,  2.4161e-02,  2.3344e-02, -1.3707e-02, -8.0652e-02,
         -1.2209e-02, -4.8322e-02,  2.3950e-02,  5.6560e-02,  4.0890e-02,
         -7.0094e-02, -4.3874e-02, -3.3973e-02, -3.6226e-02,  5.3778e-02,
          5.6594e-02,  6.0845e-02, -6.0605e-02, -5.2345e-02,  2.2961e-02,
         -4.8853e-02, -2.4157e-02,  5.3313e-04, -7.7849e-02,  1.0987e-02,
         -8.4006e-02,  1.0585e-01,  4.9765e-02, -3.2458e-01, -2.4945e-03,
          2.8307e-02, -3.7120e-02,  5.1452e-02, -6.3080e-02,  1.8275e-02,
         -3.1467e-01, -2.2925e-02,  3.9658e-02, -7.0995e-03,  1.3233e-02,
          1.0186e-01,  5.3104e-02,  6.2798e-02,  3.5321e-01,  2.9812e-02,
          1.9228e-02, -8.1668e-03, -3.5223e-02, -1.5065e-02, -9.5244e-02,
          5.1477e-03, -4.7225e-02,  1.2809e-02,  2.1152e-01, -6.2479e-02,
         -9.1506e-02,  2.9918e-02,  6.6429e-03,  2.0009e-01, -2.3685e-02,
          1.6401e-01,  3.1367e-01,  6.7967e-03, -5.3093e-02, -1.6362e-01,
          3.6276e-02, -2.5582e-02, -9.0966e-02, -5.8388e-03, -8.6487e-02,
          4.4216e-02, -3.8042e-02,  1.0169e-01, -2.5986e-02,  4.9489e-02,
         -4.6716e-02,  6.3374e-02,  2.8791e-02,  4.3376e-01, -2.5807e-02,
         -3.2225e-01,  2.9494e-02, -2.1734e-02,  5.7546e-01,  3.2926e-01,
          2.7891e-01, -5.3406e-03,  1.1175e-01]]).to(device)
        #codedict['detail'] = torch.zeros(1, 128).to(device)
        params.append(codedict)

    return params

def transform_points(points, mat):
    points = np.expand_dims(points, axis=1)
    points = cv2.transform(points, mat, points.shape)
    points = np.squeeze(points)
    return points

def print_args(parser, args):
    message = ''
    message += '----------------- Arguments ---------------\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '-------------------------------------------'
    print(message)

def main(tensor_path, pose_path, id, gpu_id, input_path):
    exp_tensor = torch.load(tensor_path)
    pose_tensor = torch.load(pose_path)
    # Figure out the device
    if gpu_id < 0:
        device = 'cpu'
    elif torch.cuda.is_available():
        if gpu_id >= torch.cuda.device_count():
            device = 'cuda:0'
        else:
            device = 'cuda:' + str(gpu_id)
    else:
        print('GPU device not available. Exit')
        exit(0)

    # Read parameters from input tensors
    src_codedicts = read_DECA_params(exp_tensor, pose_tensor, device=device)


    # Create save dirs
    nmcfs_path = os.path.join(input_path, id,  'nmfcs')
    if os.path.isdir(nmcfs_path):
        print('Conditional input files already exist!')
        shutil.rmtree(nmcfs_path)
    mkdir(nmcfs_path)
    shape_path = os.path.join(input_path, id, 'shapes')
    if os.path.isdir(shape_path):
        print('Shapes already exist')
        shutil.rmtree(shape_path)
    mkdir(shape_path)

    # run DECA decoding
    deca_cfg.model.use_tex = True
    deca = DECA(config = deca_cfg, device=device)
    original_size = (256, 256)
    tform = np.array([[ 1, 0, 0],
     [0, 1, 0],
     [ 0, 0, 1]])

    for i, src_codedict in enumerate(src_codedicts):

        opdict, visdict = deca.decode(src_codedict, device=device)

        nmfc_pth = os.path.join(nmcfs_path, str(i) + '.png')
        nmfc_image = warp(util.tensor2image(visdict['nmfcs'][0])/255, tform, output_shape=(original_size))
        nmfc_image = img_as_ubyte(nmfc_image)
        cv2.imwrite(nmfc_pth, nmfc_image)

        shape_pth = os.path.join(shape_path, str(i) + '.png')
        shape_image = warp(util.tensor2image(visdict['shape_detail_images'][0])/255, tform, output_shape=(original_size))
        shape_image = img_as_ubyte(shape_image)
        cv2.imwrite(shape_pth, shape_image)
        filename = str(i) + '.png'
        padding = '0' * (10 - len(filename))
        new_filename = padding + filename
        os.rename(os.path.join(shape_path, filename), os.path.join(shape_path, new_filename))


    print('DONE!')

if __name__=='__main__':
     

    input_path = "/home/marco/Downloads/Philipp_Eval_Neutral/inc_data"

    for subdir in os.listdir(input_path):
        sub_path = os.path.join(input_path, subdir)
        for emodir in os.listdir(sub_path):
            tensor_path = os.path.join(sub_path, emodir, "exp.pt")
            pose_path = os.path.join(sub_path, emodir, "pose.pt")
            id = emodir
            print(id)
            print(sub_path)
            main(tensor_path, pose_path, id, 0, sub_path)

    

