#import dataloader_yuv
import torch
# Git Code: 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
from argparse import ArgumentParser
#from models import HDRnetModel, Cl_HDRnetModel
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from utils import psnr, load_test_ckpt
import os
from yuv_utils import *
#from dataloader_full_only import DataLoaderX
import numpy as np
#from yuv_utils import *
from aamir_utils import *
from torchvision import datasets, models
import torchvision.transforms as transforms
from utils_color_conversion import *
import streamlit as st
from bokeh.plotting import figure
#import dataloader_full_only
#from gfxdisp_python.gfxdisp.color import *
#from gfxdisp_python.gfxdisp.pfs import pfs

def get_polynomial_matrix_fourth(full):
    full= full.squeeze()
    R = full[0].flatten().unsqueeze(1)  # shape is torch.Size([518400, 1])
    G = full[1].flatten().unsqueeze(1)
    B = full[2].flatten().unsqueeze(1)

    R_fourth = R**4
    G_fourth = G**4
    B_fourth = B**4
    
    R_cube = R**3
    G_cube = G**3
    B_cube = B**3
    
    R_sq = R**2
    G_sq = G**2
    B_sq = B**2
    
    RG = R*G
    GB = G*B
    RB = R*B
    
    ones= torch.ones((R.shape)).cuda()
    hdr_matrix= torch.cat((R_fourth, G_fourth, B_fourth, R_cube*G, R_cube*B, G_cube*R, G_cube*B, B_cube*R, B_cube*G, R_sq*G_sq, G_sq*B_sq, R_sq*B_sq, R_sq*GB, G_sq*RB, B_sq*RG,
                           R_cube, G_cube, B_cube, R*G_sq, G*B_sq,  R*B_sq, G*R_sq, B*G_sq, B*R_sq, R*G*B,
                           R_sq, G_sq, B_sq, RG, GB, RB, R, G, B, ones), dim=-1)
#    hdr_matrix= torch.cat((R_sq, G_sq, B_sq, RG, GB, RB, R, G, B, ones), dim=-1)
    
    return hdr_matrix

def get_polynomial_matrix_cube(full):
    full= full.squeeze()
    R = full[0].flatten().unsqueeze(1)  # shape is torch.Size([518400, 1])
    G = full[1].flatten().unsqueeze(1)
    B = full[2].flatten().unsqueeze(1)

    R_cube = R**3
    G_cube = G**3
    B_cube = B**3
    
    R_sq = R**2
    G_sq = G**2
    B_sq = B**2
    RG = R*G
    GB = G*B
    RB = R*B
    
    ones= torch.ones((R.shape)).cuda()
    hdr_matrix= torch.cat((R_cube, G_cube, B_cube, R*G_sq, G*B_sq,  R*B_sq, G*R_sq, B*G_sq, B*R_sq, R*G*B,
                           R_sq, G_sq, B_sq, RG, GB, RB, R, G, B, ones), dim=-1)
#    hdr_matrix= torch.cat((R_sq, G_sq, B_sq, RG, GB, RB, R, G, B, ones), dim=-1)
    
    return hdr_matrix


def get_polynomial_matrix_square(full):
    full= full.squeeze()
    R = full[0].flatten().unsqueeze(1)  # shape is torch.Size([518400, 1])
    G = full[1].flatten().unsqueeze(1)
    B = full[2].flatten().unsqueeze(1)
    
    R_sq = R**2
    G_sq = G**2
    B_sq = B**2
    RG = R*G
    GB = G*B
    RB = R*B
    
    ones= torch.ones((R.shape)).cuda()
    hdr_matrix= torch.cat((R_sq, G_sq, B_sq, RG, GB, RB, R, G, B, ones), dim=-1)
    
    return hdr_matrix
    
def get_polynomial_matrix_square_only(full):
    full= full.squeeze()
    R = full[0].flatten().unsqueeze(1)  # shape is torch.Size([518400, 1])
    G = full[1].flatten().unsqueeze(1)
    B = full[2].flatten().unsqueeze(1)
    R_sq = R**2
    G_sq = G**2
    B_sq = B**2
    ones= torch.ones((R.shape)).cuda()
    hdr_matrix= torch.cat((R_sq, G_sq, B_sq, R, G, B, ones), dim=-1)    
    return hdr_matrix    

def get_polynomial_matrix_basic(full):
    full= full.squeeze()
    R = full[0].flatten().unsqueeze(1)  # shape is torch.Size([518400, 1])
    G = full[1].flatten().unsqueeze(1)
    B = full[2].flatten().unsqueeze(1)
    ones= torch.ones((R.shape)).cuda()
    hdr_matrix= torch.cat((R, G, B, ones), dim=-1)    
    return hdr_matrix 

def get_polynomial_matrix_square_combined(full):
    full= full.squeeze()
    R = full[0].flatten().unsqueeze(1)  # shape is torch.Size([518400, 1])
    G = full[1].flatten().unsqueeze(1)
    B = full[2].flatten().unsqueeze(1)
    RG = R*G
    GB = G*B
    RB = R*B
    ones= torch.ones((R.shape)).cuda()
    hdr_matrix= torch.cat((RG, GB, RB, R, G, B, ones), dim=-1)
    
    return hdr_matrix

def parse_args():
    parser = ArgumentParser(description='HDRnet training')
    # Training, logging and checkpointing parameters
    parser.add_argument('--cuda', action='store_true', help='Use CUDA')
    parser.add_argument('--ckpt_interval', default=600, type=int, help='Interval for saving checkpoints, unit is iteration')
    parser.add_argument('--ckpt_dir', default='./ckpts_encoder_decoder_histogram', type=str, help='Checkpoint directory')
    parser.add_argument('--stats_dir', default='./stats_Polynomial_Encoding', type=str, help='Statistics directory')
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float)
    parser.add_argument('--summary_interval', default=10, type=int)
    parser.add_argument('--vgg', default=0, type=int, help='Use perception loss and which layer')
    parser.add_argument('--batch_size', default=1, type=int, help='Size of a mini-batch')
    parser.add_argument('--dataset_obj', default='Train_Dataset', type=str, help='Which dataset object to use')
    parser.add_argument('--train_num', default=None, type=int, help='Number of images to train')
    parser.add_argument('--eval_num', default=None, type=int, help='Number of images to validate')
    parser.add_argument('--test_out', default='./outputs', type=str, help='Test output path')
    # Model parameters
    parser.add_argument('--guide_cnn', action='store_true', help='Use CNN to generate guidance map')
    parser.add_argument('--guide_kernels', default=16, type=int, help='Numer of filters in guide pointwise network')
    parser.add_argument('--batch_norm', action='store_true', help='Use batch normalization')
    parser.add_argument('--input_res', default=256, type=int, help='Resolution of the down-sampled input')
    parser.add_argument('--output_res', default=(1024, 1024), type=int, nargs=2, help='Resolution of the guidemap/final output')

    parser.add_argument('--counter', default=None, type=int, help='Counter')
    parser.add_argument('--counter_validation', default=None, type=int, help='Counter')

    parser.add_argument('--vid_name', default='None', type=str, help='Which dataset/video  to use')
    parser.add_argument('--vid_name_validation', default='None', type=str, help='Which dataset/video  to use')
    
    parser.add_argument('--frame_index', default= 300, type= int, help='Which frmae to use')
    return parser.parse_args()

params = vars(parse_args())


#%%
E_in= 35
H_1= 20 # First Hidden Layer nodes/neurons
H_2 = 20 # Was 35 for the larger model, with the two lines uncommented
E_out =3 # Output dimensions
encoder= torch.nn.Sequential(
        torch.nn.Linear(E_in,H_1),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(H_1, H_2), 
        torch.nn.LeakyReLU(),
#        torch.nn.Linear(H_2, H_2), 
#        torch.nn.LeakyReLU(),
        torch.nn.Linear(H_2, E_out),
        )


state_dict, params_ = load_test_ckpt('ckpts_encoder_3_combined/Encoder_epoch_490.pt')
#params.update(params)
encoder= encoder.cuda()
encoder.load_state_dict(state_dict)
encoder.eval()

def gen_data_matrix(hdr_matrix_all):
    batch= hdr_matrix_all.shape[0]  # gives the batch size
#    img_sq = torch.squeeze(img)
#    img_flatten=  torch.flatten(hdr_matrix_all, start_dim=2)   #torch.Size([batch, 3, 518400])
    images= torch.zeros((518400*batch, 35)).cuda()
    for i in range(batch):
        img_ = hdr_matrix_all[i]   #torch.Size([518400, 3])
        images[518400*i:518400*(i+1)] = img_
        
    return images
        
def repeat_histogram(input_hist):
    batch= input_hist.shape[0]  # gives the batch size
#    img_sq = torch.squeeze(img)
   
#    img_permuted= img_flatten.T   #torch.Size([518400, 3]) , batch_size would be 518400
    
    len_feats= int(input_hist.shape[1])
    features = torch.zeros((518400*batch,len_feats)).cuda()
    
    for ii in range(batch):
        input_feats_ = input_hist[ii]
        
        features[518400*ii:518400*(ii+1),:] = input_feats_

    return features   
#%%


 
vid_dir = "../../video"
vid_name = "lego_batman"
###
fname = os.path.join( vid_dir, vid_name + "_4k", vid_name + "_960x540_420_2020_10b.yuv" )
yf_hdr = YUVFile( fname )
####
fname = os.path.join( vid_dir, vid_name + "_hd", vid_name + "_960x540_420_709_8b.yuv" )
yf_sdr = YUVFile( fname )

frame_count= yf_hdr.get_frame_count()
yf_sdr.get_frame_count()

polynomial_matrix_used= 'Encoder_3_Combined'

PSNRs=[]
#For same dataset , start from frame frame_count-300 onwards
meta_datas_lego=np.load('Meta_Datas_'+polynomial_matrix_used+'.npy')

meta_datas_first=np.load('Meta_Datas_First_'+polynomial_matrix_used+'.npy')
meta_datas_second=np.load('Meta_Datas_Second_'+polynomial_matrix_used+'.npy')
meta_datas_third=np.load('Meta_Datas_Third_'+polynomial_matrix_used+'.npy')
#avg_meta_data= torch.from_numpy(np.mean(meta_datas_lego, axis=0)).cuda()
avg_meta_data= np.mean(meta_datas_lego, axis=0)

avg_meta_first= np.mean(meta_datas_first, axis=0)*10000
avg_meta_second= np.mean(meta_datas_second, axis=0)*10000
avg_meta_third= np.mean(meta_datas_third, axis=0)*10000



att_1= avg_meta_first*-1
att_2= avg_meta_second
att_3= avg_meta_third*-1



s1= st.slider("Attribute 1", min_value= 1.0, max_value=10.0, value= 1.8 ,step=1.0 )
s2= st.slider("Attribute 2", min_value= 1.0, max_value=10.0,  value= 4.2 , step=1.0 )
s3= st.slider("Attribute 3", min_value= 1.0, max_value=10.0, value= 3.8 , step=1.0 )



i=params['frame_index']

#for i in range(frame_count) :
to_pil= transforms.ToPILImage()
  
if True:  
#    i=i+300
#    print(i)
    file_name ='frame_%06d'%(i+1)
    
    full = yf_hdr.get_frame_rgb_rec2020(i)   # in 2020 space
    full = torch.from_numpy(full.transpose((2, 0, 1)))
    full = full.cuda()#to(device)
#    save_image(torch.clamp(full, 0., 1.), 'A/In/'+file_name+'.png')                       
    hdr_matrix = get_polynomial_matrix_fourth(full) # 4th degree polynomial
    
    encoded_matrix = encoder(hdr_matrix)  # shape is torch.Size([518400, 3]) for this case
    
    
    target = yf_sdr.get_frame_rgb(i)
##    Saving rec 709 of target frames----- This works just like the matlab code
    target_xyz= rgb2xyz_m2020(target)
    target_709= xyz2rgb_709(target_xyz)
    target_709 = torch.from_numpy(target_709.transpose((2, 0, 1)))
    target_to_display= torch.clamp(target_709, 0., 1.)
##
##    save_image(target_to_display, 'A/Tar_709/'+file_name+'.png')    
#    
#    target = torch.from_numpy(target.transpose((2, 0, 1)))
#    target = target.cuda()
##    save_image(torch.clamp(target, 0., 1.), 'A/Tar/'+file_name+'.png') 
#    
#    
#                
#    target= target.squeeze()
#    target_R = target[0].flatten().unsqueeze(1)
#    target_G = target[1].flatten().unsqueeze(1)
#    target_B = target[2].flatten().unsqueeze(1)
#                
#    sdr_matrix = torch.cat((target_R, target_G, target_B), dim =-1)
                
#    meta_data = torch.linalg.lstsq(encoded_matrix, sdr_matrix).solution
#    meta_datas_lego.append(meta_data.detach().cpu().numpy())
    
    new_meta_data= avg_meta_data.astype('float32')
#    new_meta_data= np.array([[s1, s2/10, s3*-10], [s4, s5, s6], [s7*-1, s8, s9/100] ]).astype('float32')
    new= torch.from_numpy(new_meta_data).cuda()
#    
    meta_data_first= torch.from_numpy(np.array([[s1*-1]]).astype('float32')).cuda()
    meta_data_second=torch.from_numpy(np.array([[s2]]).astype('float32')).cuda()
    meta_data_third=torch.from_numpy(np.array([[s3*-1]]).astype('float32')).cuda()

    pred_first=  torch.matmul(torch.unsqueeze(encoded_matrix[:,0],1), meta_data_first/10000)
    pred_second= torch.matmul(torch.unsqueeze(encoded_matrix[:,1],1), meta_data_second/10000)
    pred_third= torch.matmul(torch.unsqueeze(encoded_matrix[:,2],1), meta_data_third/10000)
    
    pred_3_channels = torch.cat((pred_first, pred_second, pred_third), dim =-1)

    
    pred= torch.matmul(encoded_matrix, new)
    
    final_pred= pred*0.7 + pred_3_channels*0.3
    
    pred_R = final_pred[:,0].reshape(540, 960).unsqueeze(0)
    pred_G = final_pred[:,1].reshape(540, 960).unsqueeze(0)
    pred_B = final_pred[:,2].reshape(540, 960).unsqueeze(0)
    
    output= torch.cat((pred_R, pred_G, pred_B), dim=0)
#                pred_reshaped = torch.reshape(pred, (540, 960,3))
    output_clamped= torch.clamp(output, 0.0, 1.0)
    
    output_np= output_clamped.detach().cpu().numpy()
    output_np_rev= np.moveaxis(output_np, 0, -1)
    
#    output_np= im_ctrans(output_np_rev, 'rgb2020', 'srgb')
#    output_np = output_np_rev.astype('float32')
    
    xyz_op= rgb2xyz_m2020(output_np_rev)
    op= xyz2rgb_709(xyz_op)
    op_709 = torch.from_numpy(op.transpose((2, 0, 1)))
    
    
#    pfs.write_image(append(path_to_rec709, '/', frames(i).name), output_np)
#    plt.imsave('Polynomial_Diff_Movies/Encoder_Decoder_Histogram/'+polynomial_matrix_used+'/'+file_name+'.png', np.clip(op_709, 0., 1.))
    
    to_display = torch.clamp(op_709, 0., 1.)
    
    display_pil_img= to_pil(to_display)
    display_pil_target= to_pil(target_to_display)
    
    x_axis= np.linspace(0.0, 0.7, 100)
    x= full.clone()
    gray_value= np.zeros((100,))
    
    for ii in range(100):
#        print('Running loop for tone curve generation')
        x[0, :, :], x[1, :, :], x[2, :, :]= x_axis[ii],x_axis[ii],x_axis[ii]
        hdr_matrix = get_polynomial_matrix_fourth(x)
        encoded_matrix = encoder(hdr_matrix)
        
        pred_first=  torch.matmul(torch.unsqueeze(encoded_matrix[:,0],1), meta_data_first/10000)
        pred_second= torch.matmul(torch.unsqueeze(encoded_matrix[:,1],1), meta_data_second/10000)
        pred_third= torch.matmul(torch.unsqueeze(encoded_matrix[:,2],1), meta_data_third/10000)
        
        pred_3_channels = torch.cat((pred_first, pred_second, pred_third), dim =-1)
    
        
        pred= torch.matmul(encoded_matrix, new)
        
        out= pred*0.7 + pred_3_channels*0.3
        out_= ((out[0][0]+ out[0][1]+ out[0][2])/3).detach().cpu().numpy()
        gray_value[ii]= out_
    
    
    st.image(display_pil_img, 'Changed Image')
    st.image(display_pil_target, 'Ground Truth SDR Image')
    
    p = figure(title ='Tone Curve', x_axis_label= 'Input HDR (rec2020)', y_axis_label='Output SDR (rec2020)' )
    
    p.line(x_axis, gray_value)
    
    st.bokeh_chart(p, use_container_width= True)
    
#    save_image(to_display, 'A/Demo/'+file_name+'.png')            
#    
#    if i==310:
#        break
    
#    eval_psnr = psnr(output_clamped, torch.clamp(target, 0, 1)[0]).item()
#    print('PSNR is ', eval_psnr)
#    PSNRs.append(eval_psnr)
#np.save('Polynomial_Diff_Movies/Encoder_Decoder_Histogram/Meta_Datas_'+polynomial_matrix_used+'.npy', meta_datas_lego)               
#np.save('/home/am2806/public_html/ML4TMO/HDRNet/Reduced_Frames/Different_Polynomials_Lego/PSNRs_'+polynomial_matrix_used+'.npy', PSNRs)
                
