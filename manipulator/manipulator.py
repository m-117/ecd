import os
import cv2
import torch

from util.text_classifier import analyze_text
from checkpoint.checkpoint import CheckpointIO
from data.dataset import get_data_loader
from models.model import create_model

from options.base_options import BaseOptions
from util.edit_blink import add_blinking
from util.edit_pose import edit_pose_tensor
    
@torch.no_grad()
def get_style_vectors(nets, opt, trg_emotions):
    # set device to run the net on
    device = f'cuda:{opt.gpu_ids[0]}' if len(opt.gpu_ids) else 'cpu'

    # check if target emotion is valid
    if trg_emotions not in opt.selected_emotions:
        print('Invalid target emotion!')
        exit(0)

    # generate random latent vector
    z_trg = torch.randn(1, opt.latent_dim).to(device)

    # generate style vector
    y = opt.selected_emotions.index(trg_emotions)
    s_vec = nets.mapping_network(z_trg, torch.LongTensor(1).to(device).fill_(y))
    
    return s_vec

def manipulate_expressions(input_deca, input_text, global_multiplier=None, emo_override=None):
    """Analyse input text and manipulate expression, pose params for input deca params (x, 51). Returns exp tensor (x, 50), pose tensor (x, 6), actor and gt emotion list."""

    # load and set options
    opt = BaseOptions().parse()
    opt.nThreads = 1
    opt.batch_size = 1

    num_exp_coeffs = 51

    # set device to run the net on
    device = f'cuda:{opt.gpu_ids[0]}' if len(opt.gpu_ids) else 'cpu'

    # initialize models
    nets = create_model(opt)

    # load checkpoint
    ckptio = CheckpointIO(os.path.join(opt.checkpoints_dir, '{:02d}_nets_finetuned.pth'), opt, len(opt.gpu_ids)>0, **nets)
    ckptio.load(opt.which_epoch)

    # load data
    data = get_data_loader(input_deca, opt)
    data = [d.to(device) for d in data]

    # get video duration
    frames = input_deca.shape[0]

    # create neutral style vector for blendings
    neutral_style = get_style_vectors(nets, opt, "neutral")

    # run sentiment analysis on the input
    sentiment_info = analyze_text(input_text, frames)

    # create style vector for each text chunk (e.g. sentences)
    for text_chunk in sentiment_info:
        text_chunk["style"] = get_style_vectors(nets, opt, text_chunk["emotion"])
        if emo_override is not None:
            text_chunk["style"] = get_style_vectors(nets, opt, emo_override)

    # start NED routine
    with torch.no_grad():
        gaussian = cv2.getGaussianKernel(opt.seq_len,-1)
        gaussian = torch.from_numpy(gaussian).float().to(device).repeat(1,num_exp_coeffs)
        output = torch.zeros(len(data)+opt.seq_len-1, num_exp_coeffs).float().to(device)

        current_chunk = 0
        current_end_frame = sentiment_info[current_chunk]["duration"]
        
        for i, src_data in enumerate(data):
            # prepare input sequence.
            src_data = src_data.to(device)

            # update text chunk
            if i > current_end_frame:
                current_chunk += 1
                current_end_frame += sentiment_info[current_chunk]["duration"]

            # select style vector
            s_vec = sentiment_info[current_chunk]["style"]

            # neutral transition phase for emotion blending
            if int(i%current_end_frame) > (current_end_frame-10):
                s_vec = neutral_style

            # translate sequence.
            x_fake = nets.generator(src_data, s_vec)

            # add multiplier for more (or less) intense expression
            if global_multiplier is not None:
                x_fake[:, 1:51] *= global_multiplier
            x_fake[:, 1:51] *= sentiment_info[current_chunk]["multiplier"]
            

            # add sequence to output
            output[i:i+opt.seq_len] = output[i:i+opt.seq_len] + torch.squeeze(x_fake, dim=0)*gaussian
            
        # create empty pose tensor
        new_pose_tensor = torch.zeros((len(output), 6), dtype=torch.float32)

        gt_emotions = []
        actor_list = []

        for text_chunk in sentiment_info:        
            gt_emotions.append(text_chunk["emotion"])

        # add head movement for each text chunk
        current_start_frame = 0
        for text_chunk in sentiment_info:
            new_pose_tensor[current_start_frame:text_chunk["duration"], :], actor = edit_pose_tensor(torch.zeros((text_chunk["duration"], 6), dtype=torch.float32),
                                                                                                text_chunk["emotion"], "/home/marco/NED_thin/head_movement_lib.pkl")
            actor_list.append[actor]
            

        # override jaw param with output value
        new_pose_tensor[:, 3] = output[:, 0]

        # extract expression params   
        output = output.cpu().numpy()
        # add blinking animation  
        output = add_blinking(output)

        new_exp_tensor = torch.from_numpy(output[:, 1:51])

        return new_exp_tensor, new_pose_tensor, actor_list, gt_emotions
