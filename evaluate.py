import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn.functional import grid_sample
from tqdm import tqdm

import DeepLKBatch as dlk
import sift_ransac_homography as srh
from commons import setup_logging

import io
import requests
from PIL import Image, ImageFont, ImageDraw

import pdb
from sys import argv
import os
import logging
import random
import glob
from math import cos, sin, pi, sqrt
import time
import sys
import gc
import argparse

# USAGE:
# python3 evaluate.py MODE FOLDER_NAME DATAPATH MODEL_PATH VGG_MODEL_PATH --TEST_DATA_SAVE_PATH

# TRAIN:
# python3 evaluate.py train woodbridge ../sat_data/ trained_model_output.pth ../models/vgg16_model.pth

# TEST:
# python3 evaluate.py test woodbridge ../sat_data/ ../models/conv_02_17_18_1833.pth ../models/vgg16_model.pth -t test_out.txt

MODE = None
FOLDER_NAME = None
DATA_PATH = None
MODEL_PATH = None
VGG_MODEL_PATH = None

# size scale range
min_scale = 0.75
max_scale = 1.25

# rotation range (-angle_range, angle_range)
angle_range = 15 # degrees

# projective variables (p7, p8)
projective_range = 0

# translation (p3, p6)
translation_range = 10 # pixels

# possible segment sizes
lower_sz = 200 # pixels, square
upper_sz = 220

# amount to pad when cropping segment, as ratio of size, on all 4 sides
warp_pad = 0.4

# normalized size of all training pairs
training_sz = 175
training_sz_pad = round(training_sz + training_sz * 2 * warp_pad)

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

def random_param_generator():
    # create random ground truth warp parameters in the specified ranges
    scale = random.uniform(min_scale, max_scale)
    angle = random.uniform(-angle_range, angle_range)
    projective_x = random.uniform(-projective_range, projective_range)
    projective_y = random.uniform(-projective_range, projective_range)
    translation_x = random.uniform(-translation_range, translation_range)
    translation_y = random.uniform(-translation_range, translation_range)
    rad_ang = angle / 180 * pi

    p_gt = Variable(torch.Tensor([scale + cos(rad_ang) - 2,
                              -sin(rad_ang),
                              translation_x,
                              sin(rad_ang),
                              scale + cos(rad_ang) - 2,
                              translation_y,
                              projective_x,
                              projective_y])).to(DEVICE)

    p_gt = p_gt.view(8,1)
    p_gt = p_gt.repeat(1,1,1)

    return p_gt


def static_data_generator(batch_size):
    # similar to data_generator, except with static warp parameters (easier for testing)
    images_dir = glob.glob(DATA_PATH)
    random.shuffle(images_dir)

    img = Image.open(images_dir[0])
    template = Image.open(images_dir[1])

    in_W, in_H = img.size

    img_batch = Variable(torch.zeros(batch_size, 3, training_sz, training_sz)).to(DEVICE)
    template_batch = Variable(torch.zeros(batch_size, 3, training_sz, training_sz)).to(DEVICE)
    param_batch = Variable(torch.zeros(batch_size, 8, 1)).to(DEVICE)

    for i in range(batch_size):

        # randomly choose size and top left corner of image for sampling
        seg_sz = 200
        seg_sz_pad = round(seg_sz + seg_sz * 2 * warp_pad)

        loc_x = 40
        loc_y = 40

        img_seg_pad = img.crop((loc_x, loc_y, loc_x + seg_sz_pad, loc_y + seg_sz_pad))
        img_seg_pad = img_seg_pad.resize((training_sz_pad, training_sz_pad))

        template_seg_pad = template.crop((loc_x, loc_y, loc_x + seg_sz_pad, loc_y + seg_sz_pad))
        template_seg_pad = template_seg_pad.resize((training_sz_pad, training_sz_pad))

        img_seg_pad = Variable(transforms.ToTensor()(img_seg_pad)).to(DEVICE)
        template_seg_pad = Variable(transforms.ToTensor()(template_seg_pad)).to(DEVICE)

        scale = 1.2
        angle = 10
        projective_x = 0
        projective_y = 0
        translation_x = -5
        translation_y = 10

        rad_ang = angle / 180 * pi

        p_gt = Variable(torch.Tensor([scale + cos(rad_ang) - 2,
                                      -sin(rad_ang),
                                      translation_x,
                                      sin(rad_ang),
                                      scale + cos(rad_ang) - 2,
                                      translation_y,
                                      projective_x,
                                      projective_y])).to(DEVICE)

        p_gt = p_gt.view(8,1)
        p_gt = p_gt.repeat(1,1,1)

        inv_func = dlk.InverseBatch()

        img_seg_pad_w, _ = dlk.warp_hmg(img_seg_pad.unsqueeze(0), dlk.H_to_param(inv_func(dlk.param_to_H(p_gt))))

        img_seg_pad_w.squeeze_(0)

        pad_side = round(training_sz * warp_pad)

        img_seg_w = img_seg_pad_w[:,
                    pad_side : pad_side + training_sz,
                    pad_side : pad_side + training_sz]



        template_seg = template_seg_pad[:,
                       pad_side : pad_side + training_sz,
                       pad_side : pad_side + training_sz]

        img_batch[i, :, :, :] = img_seg_w[0:3,:,:]
        template_batch[i, :, :, :] = template_seg[0:3,:,:]

        param_batch[i, :, :] = p_gt[0, :, :].data

    return img_batch, template_batch, param_batch


def data_generator(batch_size):
    """
    create batch of normalized training pairs

    batch_size [in, int] : number of pairs
    img_batch [out, Tensor N x 3 x training_sz x training_sz] : batch of images
    template_batch [out, Tensor N x 3 x training_sz x training_sz] : batch of templates
    param_batch [out, Tensor N x 8 x 1] : batch of ground truth warp parameters
    """

    # randomly choose 2 aligned images
    images_dir = glob.glob(DATA_PATH)
    random.shuffle(images_dir)

    img = Image.open(images_dir[0])
    template = Image.open(images_dir[1])

    in_W, in_H = img.size

    # initialize output tensors
    img_batch = Variable(torch.zeros(batch_size, 3, training_sz, training_sz)).to(DEVICE)
    template_batch = Variable(torch.zeros(batch_size, 3, training_sz, training_sz)).to(DEVICE)
    param_batch = Variable(torch.zeros(batch_size, 8, 1)).to(DEVICE)

    for i in range(batch_size):
        # randomly choose size and top left corner of image for sampling
        seg_sz = random.randint(lower_sz, upper_sz)
        seg_sz_pad = round(seg_sz + seg_sz * 2 * warp_pad)

        loc_x = random.randint(0, (in_W - seg_sz_pad) - 1)
        loc_y = random.randint(0, (in_H - seg_sz_pad) - 1)

        img_seg_pad = img.crop((loc_x, loc_y, loc_x + seg_sz_pad, loc_y + seg_sz_pad))
        img_seg_pad = img_seg_pad.resize((training_sz_pad, training_sz_pad))

        template_seg_pad = template.crop((loc_x, loc_y, loc_x + seg_sz_pad, loc_y + seg_sz_pad))
        template_seg_pad = template_seg_pad.resize((training_sz_pad, training_sz_pad))

        img_seg_pad = Variable(transforms.ToTensor()(img_seg_pad)).to(DEVICE)
        template_seg_pad = Variable(transforms.ToTensor()(template_seg_pad)).to(DEVICE)

        # create random ground truth
        scale = random.uniform(min_scale, max_scale)
        angle = random.uniform(-angle_range, angle_range)
        projective_x = random.uniform(-projective_range, projective_range)
        projective_y = random.uniform(-projective_range, projective_range)
        translation_x = random.uniform(-translation_range, translation_range)
        translation_y = random.uniform(-translation_range, translation_range)

        rad_ang = angle / 180 * pi

        p_gt = Variable(torch.Tensor([scale + cos(rad_ang) - 2,
                                      -sin(rad_ang),
                                      translation_x,
                                      sin(rad_ang),
                                      scale + cos(rad_ang) - 2,
                                      translation_y,
                                      projective_x,
                                      projective_y])).to(DEVICE)

        p_gt = p_gt.view(8,1)
        p_gt = p_gt.repeat(1,1,1)

        inv_func = dlk.InverseBatch()

        img_seg_pad_w, _ = dlk.warp_hmg(img_seg_pad.unsqueeze(0), dlk.H_to_param(inv_func(dlk.param_to_H(p_gt))))

        img_seg_pad_w.squeeze_(0)

        pad_side = round(training_sz * warp_pad)

        img_seg_w = img_seg_pad_w[:,
                    pad_side : pad_side + training_sz,
                    pad_side : pad_side + training_sz]

        template_seg = template_seg_pad[:,
                       pad_side : pad_side + training_sz,
                       pad_side : pad_side + training_sz]

        img_batch[i, :, :, :] = img_seg_w
        template_batch[i, :, :, :] = template_seg

        param_batch[i, :, :] = p_gt[0, :, :].data

    return img_batch, template_batch, param_batch


def corner_loss(p, p_gt):
    """
    p [in, torch tensor] : batch of regressed warp parameters
    p_gt [in, torch tensor] : batch of gt warp parameters
    loss [out, float] : sum of corner loss over minibatch
    """
    batch_size, _, _ = p.size()

    # compute corner loss
    H_p = dlk.param_to_H(p)
    H_gt = dlk.param_to_H(p_gt)

    corners = Variable(torch.Tensor([[-training_sz_pad/2, training_sz_pad/2, training_sz_pad/2, -training_sz_pad/2],
                                     [-training_sz_pad/2, -training_sz_pad/2, training_sz_pad/2, training_sz_pad/2],
                                     [1, 1, 1, 1]])).to(DEVICE)

    corners = corners.repeat(batch_size, 1, 1)
    corners_w_p = H_p.bmm(corners)
    corners_w_gt = H_gt.bmm(corners)
    corners_w_p = corners_w_p[:, 0:2, :] / corners_w_p[:, 2:3, :]
    corners_w_gt = corners_w_gt[:, 0:2, :] / corners_w_gt[:, 2:3, :]

    loss = ((corners_w_p - corners_w_gt) ** 2).sum()

    return loss

def test():
    dlk_vgg16 = dlk.DeepLK(dlk.vgg16Conv(VGG_MODEL_PATH)).to(DEVICE)
    dlk_trained = dlk.DeepLK(dlk.custom_net(MODEL_PATH)).to(DEVICE)

    testbatch_sz = 1 # keep as 1 in order to compute corner error accurately
    test_rounds_num = 50
    rounds_per_pair = 50

    test_results = np.zeros((test_rounds_num, 5), dtype=float)

    logging.info(f"Testing...\n save testing data path: {TEST_DATA_SAVE_PATH}, data path: {DATA_PATH}, " \
                 f"model path: {MODEL_PATH}, device: {DEVICE}, min_scale: {min_scale}, max_scale: {max_scale}" \
                 f"angle range: {angle_range}, projective range: {projective_range}, translation range: {translation_range}" \
                 f"lower size: {lower_sz}, upper size: {upper_sz}, warp pad: {warp_pad}, testing batch size: {testbatch_sz}"\
                 f"number of testing rounds: {test_rounds_num}, rounds per pair: {rounds_per_pair}")

    img_test_data = Variable(torch.zeros(test_rounds_num, 3, training_sz, training_sz)).to(DEVICE)
    template_test_data = Variable(torch.zeros(test_rounds_num, 3, training_sz, training_sz)).to(DEVICE)
    param_test_data = Variable(torch.zeros(test_rounds_num, 8, 1)).to(DEVICE)

    for i in tqdm(range(round(test_rounds_num / rounds_per_pair))):
        batch_index = i * rounds_per_pair

        img_batch, template_batch, param_batch = data_generator(rounds_per_pair)

        img_test_data[batch_index:batch_index + rounds_per_pair, :, :, :] = img_batch
        template_test_data[batch_index:batch_index + rounds_per_pair, :, :, :] = template_batch
        param_test_data[batch_index:batch_index + rounds_per_pair, :, :] = param_batch

        time.sleep(0.01)

    for i in range(test_rounds_num):
        img_batch_unnorm = img_test_data[i, :, :, :].unsqueeze(0)
        template_batch_unnorm = template_test_data[i, :, :, :].unsqueeze(0)
        param_batch = param_test_data[i, :, :].unsqueeze(0)

        img_batch = dlk.normalize_img_batch(img_batch_unnorm)
        template_batch = dlk.normalize_img_batch(template_batch_unnorm)

        img_batch_coarse = nn.AvgPool2d(4)(img_batch)
        template_batch_coarse = nn.AvgPool2d(4)(template_batch)

        vgg_param, _ = dlk_vgg16(img_batch, template_batch, tol=1e-3, max_itr=70, conv_flag=1)
        trained_param, _  = dlk_trained(img_batch, template_batch, tol=1e-3, max_itr=70, conv_flag=1)
        coarse_param, _ = dlk_vgg16(img_batch_coarse, template_batch_coarse, tol=1e-3, max_itr=70, conv_flag=0)

        vgg_loss = corner_loss(vgg_param, param_batch)
        no_op_loss = corner_loss(Variable(torch.zeros(testbatch_sz, 8, 1)), param_batch).to(DEVICE)

        trained_loss = corner_loss(trained_param, param_batch)
        coarse_loss = corner_loss(coarse_param, param_batch)

        # srh_param = srh.get_param(img_batch_unnorm, template_batch_unnorm, training_sz)
        # no_op_loss = corner_loss(Variable(torch.zeros(testbatch_sz, 8, 1)), param_batch)
        # srh_loss = corner_loss(srh_param, param_batch)

        test_results[i, 0] = vgg_loss
        test_results[i, 1] = trained_loss
        test_results[i, 2] = coarse_loss
        test_results[i, 3] = no_op_loss
        # test_results[i, 4] = srh_loss

        print('test: ', i,
              ' vgg16 loss: ', round(sqrt(float(vgg_loss)/4),2),
              ' trained loss: ', round(sqrt(float(trained_loss)/4),2),
              ' coarse pix loss: ', round(sqrt(float(coarse_loss)/4),2),
              ' no-op loss: ', round(sqrt(float(no_op_loss)/4),2))
        # ' srh loss: ', round(sqrt(float(srh_loss)/4),2))

        sys.stdout.flush()

    #### --- Visualize Testing

    # warped_back_custom, _ = dlk.warp_hmg(img_batch_unnorm, trained_param)
    # warped_back_srh, _ = dlk.warp_hmg(img_batch_unnorm, srh_param)
    # warped_back_vgg, _ = dlk.warp_hmg(img_batch_unnorm, vgg_param)
    # warped_back_iclk, _ = dlk.warp_hmg(img_batch_unnorm, coarse_param)

    # transforms.ToPILImage()(img_batch_unnorm[0,:,:,:].data).show()
    # time.sleep(0.25)
    # transforms.ToPILImage()(template_batch_unnorm[0,:,:,:].data).show()
    # time.sleep(0.25)
    # transforms.ToPILImage()(warped_back_custom[0,:,:,:].data).show()
    # time.sleep(0.25)
    # transforms.ToPILImage()(warped_back_srh[0,:,:,:].data).show()
    # time.sleep(0.25)
    # transforms.ToPILImage()(warped_back_vgg[0,:,:,:].data).show()
    # time.sleep(0.25)
    # transforms.ToPILImage()(warped_back_iclk[0,:,:,:].data).show()

    # pdb.set_trace()

    #### ---

    np.savetxt(TEST_DATA_SAVE_PATH, test_results, delimiter=',')



def train():
    best_valid_loss = 0
    minibatch_sz = 10
    num_minibatch = 25000
    valid_batch_sz = 10
    valid_num_generator = 50

    logging.info(f"Training...\n data path: {DATA_PATH}, " \
                 f"model path: {MODEL_PATH}, device: {DEVICE}, min_scale: {min_scale}, max_scale: {max_scale}" \
                 f"angle range: {angle_range}, projective range: {projective_range}, translation range: {translation_range}" \
                 f"lower size: {lower_sz}, upper size: {upper_sz}, warp pad: {warp_pad}, minibatch size: {minibatch_sz}"\
                 f"number of mini batches: {num_minibatch}")

    img_train_data = Variable(torch.zeros(num_minibatch, 3, training_sz, training_sz)).to(DEVICE)
    template_train_data = Variable(torch.zeros(num_minibatch, 3, training_sz, training_sz)).to(DEVICE)
    param_train_data = Variable(torch.zeros(num_minibatch, 8, 1)).to(DEVICE)

    for i in tqdm(range(round(num_minibatch / minibatch_sz))):
        batch_index = i * minibatch_sz
        img_batch, template_batch, param_batch = data_generator(minibatch_sz)
        img_train_data[batch_index:batch_index + minibatch_sz, :, :, :] = img_batch
        template_train_data[batch_index:batch_index + minibatch_sz, :, :, :] = template_batch
        param_train_data[batch_index:batch_index + minibatch_sz, :, :] = param_batch

    dlk_net = dlk.DeepLK(dlk.vgg16Conv(VGG_MODEL_PATH)).to(DEVICE)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, dlk_net.conv_func.parameters()), lr=0.0001)

    valid_img_batch = Variable(torch.zeros(valid_batch_sz * valid_num_generator, 3, training_sz, training_sz)).to(DEVICE)
    valid_template_batch = Variable(torch.zeros(valid_batch_sz * valid_num_generator, 3, training_sz, training_sz)).to(DEVICE)
    valid_param_batch = Variable(torch.zeros(valid_batch_sz * valid_num_generator, 8, 1)).to(DEVICE)

    for i in range(valid_num_generator):
        print('gathering validation data...', i+1, ' / ', valid_num_generator)
        valid_img_batch[i * valid_batch_sz: i * valid_batch_sz + valid_batch_sz,:,:,:], valid_template_batch[i * valid_batch_sz: i * valid_batch_sz + valid_batch_sz,:,:,:], valid_param_batch[i * valid_batch_sz: i * valid_batch_sz + valid_batch_sz,:,:] = data_generator(valid_batch_sz)


    for i in range(num_minibatch):
        start = time.time()
        optimizer.zero_grad()

        img_batch_unnorm = img_train_data[i, :, :, :].unsqueeze(0)
        template_batch_unnorm = template_train_data[i, :, :, :].unsqueeze(0)
        training_param_batch = param_train_data[i, :, :].unsqueeze(0)

        training_img_batch = dlk.normalize_img_batch(img_batch_unnorm)
        training_template_batch = dlk.normalize_img_batch(template_batch_unnorm)

        # 	forward pass of training minibatch through dlk
        dlk_param_batch, _ = dlk_net(training_img_batch, training_template_batch, tol=1e-3, max_itr=1, conv_flag=1)

        loss = corner_loss(dlk_param_batch, training_param_batch)

        loss.backward()

        optimizer.step()

        dlk_valid_param_batch, _ = dlk_net(valid_img_batch, valid_template_batch, tol=1e-3, max_itr=1, conv_flag=1)

        valid_loss = corner_loss(dlk_valid_param_batch, valid_param_batch)

        print('mb: ', i+1, ' training loss: ', float(loss), ' validation loss: ', float(valid_loss), end='')

        if (i == 0) or (float(valid_loss) < float(best_valid_loss)):
            best_valid_loss = valid_loss
            torch.save(dlk_net.conv_func, MODEL_PATH)
            print(' best validation loss: ', float(best_valid_loss), ' (saving)')
        else:
            print(' best validation loss: ', float(best_valid_loss))

        gc.collect()
        sys.stdout.flush()

        end = time.time()

# print('elapsed: ', end-start)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode")
    parser.add_argument("--datasets_folder")
    parser.add_argument("--model_path")
    parser.add_argument("--vgg_model_path")
    parser.add_argument("--testing_data_dave_path")

    args = parser.parse_args()

    MODE = args.mode
    DATASETS_FOLDER = args.datasets_folder
    DATA_PATH = f"{DATASETS_FOLDER}/{FOLDER_NAME}/images/*.png"
    MODEL_PATH = args.model_path
    VGG_MODEL_PATH = args.vgg_model_path
    TEST_DATA_SAVE_PATH = args.testing_data_dave_path

    setup_logging("logs")

    if MODE == 'test':
        if args.TEST_DATA_SAVE_PATH == None:
            exit('Must supply TEST_DATA_SAVE_PATH argument in test mode')
        else:
            TEST_DATA_SAVE_PATH = args.TEST_DATA_SAVE_PATH
            test()
    else:
        train()










