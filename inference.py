import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from yuv_utils import *
import model_All_in_One as model
from utils import psnr, load_test_ckpt
from utils_color_conversion import *
from torchvision.utils import save_image
import torchvision.transforms as transforms
import streamlit as st
import matplotlib.pyplot as plt
from argparse import ArgumentParser

cinn = model.cINN(0)

state_dict = {k: v for k, v in torch.load('checkpoints/ckpts_jungles_AIO/CINN_epoch_75.pt').items() if
              'tmp_var' not in k}
cinn.load_state_dict(state_dict)
cinn.cuda()
cinn.eval()


def parse_args():
    parser = ArgumentParser(description='Streamlit for 3 dimensions')
    # Training, logging and checkpointing parameters
    parser.add_argument('--cuda', action='store_true', help='Use CUDA')

    parser.add_argument('--frame_index', default=300, type=int, help='Which frmae to use')
    return parser.parse_args()


params = vars(parse_args())


###
# fname = os.path.join( vid_dir, vid_name + "_4k", vid_name + "_960x540_420_2020_10b.yuv" )
# yf_hdr = YUVFile( fname )
#####
# fname = os.path.join( vid_dir, vid_name + "_hd", vid_name + "_960x540_420_709_8b.yuv" )
# yf_sdr = YUVFile( fname )
#
# yf_hdr.get_frame_count()
# yf_sdr.get_frame_count()

def get_rgb(full):
    full = full.squeeze()
    R = full[0].flatten().unsqueeze(1)  # shape is torch.Size([518400, 1])
    G = full[1].flatten().unsqueeze(1)
    B = full[2].flatten().unsqueeze(1)

    matrix = torch.cat((R, G, B), dim=-1)

    return matrix


def get_polynomial_matrix_fourth(full):
    full = full.squeeze()
    R = full[0].flatten().unsqueeze(1)  # shape is torch.Size([518400, 1])
    G = full[1].flatten().unsqueeze(1)
    B = full[2].flatten().unsqueeze(1)

    R_fourth = R ** 4
    G_fourth = G ** 4
    B_fourth = B ** 4

    R_cube = R ** 3
    G_cube = G ** 3
    B_cube = B ** 3

    R_sq = R ** 2
    G_sq = G ** 2
    B_sq = B ** 2

    RG = R * G
    GB = G * B
    RB = R * B

    ones = torch.ones((R.shape)).cuda()
    hdr_matrix = torch.cat((R_fourth, G_fourth, B_fourth, R_cube * G, R_cube * B, G_cube * R, G_cube * B, B_cube * R,
                            B_cube * G, R_sq * G_sq, G_sq * B_sq, R_sq * B_sq, R_sq * GB, G_sq * RB, B_sq * RG,
                            R_cube, G_cube, B_cube, R * G_sq, G * B_sq, R * B_sq, G * R_sq, B * G_sq, B * R_sq,
                            R * G * B,
                            R_sq, G_sq, B_sq, RG, GB, RB, R, G, B), dim=-1)
    #    hdr_matrix= torch.cat((R_sq, G_sq, B_sq, RG, GB, RB, R, G, B, ones), dim=-1)

    return hdr_matrix


# %%

# /anfs/gfxdisp/am2806/video
vid_dir = "/anfs/gfxdisp/am2806/video"
vid_name = "planet_earth_e3_jungles"
###
fname = os.path.join(vid_dir, vid_name + "_4k", vid_name + "_960x540_420_2020_10b.yuv")
yf_hdr = YUVFile(fname)
####
fname = os.path.join(vid_dir, vid_name + "_hd", vid_name + "_960x540_420_709_8b.yuv")
yf_sdr = YUVFile(fname)

frame_count = yf_hdr.get_frame_count()
yf_sdr.get_frame_count()

# index=[300]
# z_from_training_set=[]
#
# att_1= -0.08435339
# att_2=-0.0075515
# att_3= 0.0111832   # This remains fixed
#
#
# s1= st.slider("Latent code 1", min_value= -1.0, max_value=1.0, value= 0.84 ,step=0.02 )
# s2= st.slider("Latent code 2", min_value= -1.0, max_value=1.0,  value= 0.08 , step=0.02 )
# s3= st.slider("Latent code 3", min_value= -1.0, max_value=1.0,  value= 0.12 , step=0.02 )
#
# i = params['frame_index']
# to_pil= transforms.ToPILImage()
#
#
#
# if True :
#
#
#    i=i
#
#    file_name ='frame_%06d'%(i+1)
#
#    full = yf_hdr.get_frame_rgb_rec2020(i)   # in 2020 space
#    full = torch.from_numpy(full.transpose((2, 0, 1)))
#    full = full.cuda()#to(device)
#
##    hdr_frame_luminance= 0.212656*full[0,:,:] + 0.715158*full[1,:,:] + 0.072186*full[2,:,:]
##    hist1 = SingleDimHistLayer()(hdr_frame_luminance.unsqueeze(0))
##
##    l= torch.cat((get_polynomial_matrix_fourth(full), torch.cat([hist1]*518400)), 1)
#
#    l= get_polynomial_matrix_fourth(full)
#
#    target_np = yf_sdr.get_frame_rgb(i)
#    target = torch.from_numpy(target_np.transpose((2, 0, 1)))
#
##    x= get_rgb(target.cuda())
##
##    z, log_j = cinn(x, l)
#    z_new= np.array([s1/-10, s2/-10, s3/10]).astype('float32')
#
##    z_new = np.array([att_1, att_2, att_3]).astype('float32')
#    new= torch.from_numpy(z_new)
#
#    z_rand= new.unsqueeze(0)
#    z= torch.cat([z_rand]*518400).cuda()
#
#
#    target_xyz= rgb2xyz_m2020(target_np)
#    target_709= xyz2rgb_709(target_xyz)
#    target_709 = torch.from_numpy(target_709.transpose((2, 0, 1)))
#
#
#    with torch.no_grad():
#        pred, jac_rev = cinn.reverse_sample(z, l)
#
#    pred_R = pred[:,0].reshape(540, 960).unsqueeze(0)
#    pred_G = pred[:,1].reshape(540, 960).unsqueeze(0)
#    pred_B = pred[:,2].reshape(540, 960).unsqueeze(0)
#
#    output= torch.cat((pred_R, pred_G, pred_B), dim=0)
#    output_clamped= torch.clamp(output, 0.0, 1.0)
#    output_np= output_clamped.detach().cpu().numpy()
#    output_np_rev= np.moveaxis(output_np, 0, -1)
#
##    output_np= im_ctrans(output_np_rev, 'rgb2020', 'srgb')
##    output_np = output_np_rev.astype('float32')
#
#    xyz_op= rgb2xyz_m2020(output_np_rev)
#    op= xyz2rgb_709(xyz_op)
#    op_709 = torch.from_numpy(op.transpose((2, 0, 1)))
#
#    eval_psnr = psnr(torch.clamp(op_709.cuda(), 0., 1.), torch.clamp(target_709.cuda(), 0., 1.)).item()
#    print('PSNR is Averaged', eval_psnr)
#
#
#    display_pil_img= to_pil(torch.clamp(op_709.cuda(), 0., 1.))
#    display_pil_target= to_pil(torch.clamp(target_709.cuda(), 0., 1.))
#
#    st.image(display_pil_img, 'Changed Image')
#    st.write('PSNR: ', eval_psnr)
#    st.image(display_pil_target, 'Ground Truth SDR Image')

# %%
index = [100, 150, 200, 250, 300, 350, 400]

index = [188]
for i in index:
    print(i)

    #    if i ==500:
    #        break
    file_name = 'frame_%06d' % (i + 1)

    full = yf_hdr.get_frame_rgb_rec2020(i)  # in 2020 space
    full = torch.from_numpy(full.transpose((2, 0, 1)))
    full = full.cuda()  # to(device)

    #    hdr_frame_luminance= 0.212656*full[0,:,:] + 0.715158*full[1,:,:] + 0.072186*full[2,:,:]
    #    hist1 = SingleDimHistLayer()(hdr_frame_luminance.unsqueeze(0))
    #
    #    l= torch.cat((get_polynomial_matrix_fourth(full), torch.cat([hist1]*518400)), 1)

    l = get_polynomial_matrix_fourth(full)

    target_np = yf_sdr.get_frame_rgb(i)
    target = torch.from_numpy(target_np.transpose((2, 0, 1)))

    x = get_rgb(target.cuda())

    z, log_j = cinn(x, l)

    with torch.no_grad():
        pred, jac_rev = cinn.reverse_sample(z, l)

    pred_R = pred[:, 0].reshape(540, 960).unsqueeze(0)
    pred_G = pred[:, 1].reshape(540, 960).unsqueeze(0)
    pred_B = pred[:, 2].reshape(540, 960).unsqueeze(0)

    output = torch.cat((pred_R, pred_G, pred_B), dim=0)
    output_clamped = torch.clamp(output, 0.0, 1.0)
    output_np = output_clamped.detach().cpu().numpy()
    output_np_rev = np.moveaxis(output_np, 0, -1)

    #    output_np= im_ctrans(output_np_rev, 'rgb2020', 'srgb')
    #    output_np = output_np_rev.astype('float32')

    xyz_op = rgb2xyz_m2020(output_np_rev)
    op = xyz2rgb_709(xyz_op)
    op_709 = torch.from_numpy(op.transpose((2, 0, 1)))

    target_xyz = rgb2xyz_m2020(target_np)
    target_709 = xyz2rgb_709(target_xyz)
    target_709 = torch.from_numpy(target_709.transpose((2, 0, 1)))

    eval_psnr = psnr(torch.clamp(op_709.cuda(), 0., 1.), torch.clamp(target_709.cuda(), 0., 1.)).item()
    print('PSNR is ', eval_psnr)

    av = torch.mean(z, axis=0)
    print(av)
    #    z_from_training_set.append(av.detach().cpu().numpy())

    av = torch.tensor(np.array([4, -0.2, 0.4])) / 10
    av = av.type(torch.FloatTensor)
    z_rand = av.unsqueeze(0)
    z = torch.cat([z_rand] * 518400).cuda()

    with torch.no_grad():
        pred, jac_rev = cinn.reverse_sample(z, l)

    pred_R = pred[:, 0].reshape(540, 960).unsqueeze(0)
    pred_G = pred[:, 1].reshape(540, 960).unsqueeze(0)
    pred_B = pred[:, 2].reshape(540, 960).unsqueeze(0)

    output = torch.cat((pred_R, pred_G, pred_B), dim=0)
    output_clamped = torch.clamp(output, 0.0, 1.0)
    output_np = output_clamped.detach().cpu().numpy()
    output_np_rev = np.moveaxis(output_np, 0, -1)

    #    output_np= im_ctrans(output_np_rev, 'rgb2020', 'srgb')
    #    output_np = output_np_rev.astype('float32')

    xyz_op = rgb2xyz_m2020(output_np_rev)
    op = xyz2rgb_709(xyz_op)
    op_709 = torch.from_numpy(op.transpose((2, 0, 1)))

    eval_psnr = psnr(torch.clamp(op_709.cuda(), 0., 1.), torch.clamp(target_709.cuda(), 0., 1.)).item()

    #    save_image(torch.clamp(target_709.cuda(), 0., 1.), 'Samples/Target.png')
    #    save_image(torch.clamp(op_709.cuda(), 0., 1.), 'Samples/Output_grid_zeta.png')
    print('PSNR is Averaged', eval_psnr)
#
#
