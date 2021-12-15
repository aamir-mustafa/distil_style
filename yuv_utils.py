import numpy as np
import os.path
import re
import math
import imutils
from pfs import pfs
import matplotlib.pyplot as plt
def decode_video_props( fname ):

    vprops = dict()
    vprops["width"]=1920
    vprops["height"]=1080

    vprops["fps"] = 24
    vprops["bit_depth"] = 8
    vprops["color_space"] = '2020'
    vprops["chroma_ss"] = '420'

    bname = os.path.splitext(os.path.basename(fname))[0]
    fp = bname.split("_")

    res_match = re.compile( '(\d+)x(\d+)p?' )

    for field in fp:

        if res_match.match( field ):
            res = field.split( "x")
            if len(res) != 2:
                raise ValueError("Cannot decode the resolution")
            vprops["width"]=int(res[0])
            vprops["height"]=int(res[1])
            continue

        if field=="444" or field=="420":
            vprops["chroma_ss"]=field

        if field=="10" or field=="10b":
            vprops["bit_depth"]=10

        if field=="8" or field=="8b":
            vprops["bit_depth"]=8

        if field=="2020" or field=="709":
            vprops["color_space"]=field

        if field=="bt709":
            vprops["color_space"]="709"

        if field=="ct2020" or field=="pq2020":
            vprops["color_space"]="2020"

    return vprops

def convert420to444(plane):

    #TODO: Replace with a proper filter
    return imutils.resize( plane, width=plane.shape[1]*2, height=plane.shape[0]*2 )


def fixed2float(YUV_shifted, bits):
    offset = 16/219
    weight = 1/(2**(bits-8)*219)

    YUV = np.empty(YUV_shifted.shape, dtype=np.float32)
    YUV[..., 0] = np.clip(weight*YUV_shifted[..., 0].astype(np.float32) - offset, 0, 1)

    offset = 128/224
    weight = 1/(2**(bits-8)*224)

    YUV[..., 1] = np.clip(weight*YUV_shifted[..., 1].astype(np.float32) - offset, -0.5, 0.5)
    YUV[..., 2] = np.clip(weight*YUV_shifted[..., 2].astype(np.float32) - offset, -0.5, 0.5)

    return YUV


rgb2ycbcr = np.array([[0.2126, 0.7152, 0.0722],
                      [-0.114572, -0.385428, 0.5],
                      [0.5, -0.454153, -0.045847]], dtype=np.float32)

ycbcr2rgb = np.array([[1, 0, 1.5748],
                      [1, -0.18733, -0.46813],
                      [1, 1.85563, 0]], dtype=np.float32)   # This is rec 709 space
    
    
ycbcr2rgb_rec2020 = np.array([[1, 0, 1.47460],
                      [1, -0.16455, -0.57135],
                      [1, 1.88140, 0]], dtype=np.float32)   # This is rec 2020 space, This is to be employed
    # on 4k video yuv file


rgb709_xyz = np.array([[0.4124, 0.3576, 0.1805],
                      [0.2126, 0.7152, 0.0722],
                      [0.0193, 0.1192, 0.9505]], dtype=np.float32)


rgb2020_xyz = np.array([[0.6370, 0.1446, 0.1689],
                      [0.2627, 0.6780, 0.0593],
                      [0.0000, 0.0281, 1.0610]], dtype=np.float32)     


xyz2rgb709 =   np.array([[3.2406, -1.5372, -0.4986],
                      [-0.9689,  1.8758,  0.0415],
                      [0.0557, -0.2040,  1.0570]], dtype=np.float32)      

xyz2rgb2020 =   np.linalg.inv (
                        np.array([[0.6370, 0.1446, 0.1689],
                      [0.2627, 0.6780, 0.0593],
                      [0.0000, 0.0281, 1.0610]], dtype=np.float32) )
    
                   
xyz = np.matmul(rgb709_xyz , ycbcr2rgb)

rgb_2020 = np.matmul(xyz2rgb2020 , xyz)   # This is to be employed on hd yuv file
                    
class YUVFile:

    def __init__(self, file_name):        
        self.file_name = file_name

        if not os.path.isfile(file_name):
            raise FileNotFoundError( "File {} not found".format(file_name) )

        vprops = decode_video_props(file_name)
#        print(vprops)

        self.bit_depth = vprops["bit_depth"]
        self.frame_bytes = int(vprops["width"]*vprops["height"])
        self.y_pixels = int(self.frame_bytes)
        self.y_shape = (vprops["height"], vprops["width"])

        if vprops["chroma_ss"]=="444":
            self.frame_bytes *= 3
            self.uv_pixels = self.y_pixels
            self.uv_shape = self.y_shape
        else: # Chroma sub-sampling
            self.frame_bytes = self.frame_bytes*3/2
            self.uv_pixels = int(self.y_pixels/4)
            self.uv_shape = (int(self.y_shape[0]/2), int(self.y_shape[1]/2))

        self.frame_pixels = self.frame_bytes
        if vprops["bit_depth"]>8:
            self.frame_bytes *= 2
            self.dtype = np.uint16
        else:
            self.dtype = np.uint8


        self.frame_count = os.stat(file_name).st_size / self.frame_bytes
        if math.ceil(self.frame_count)!=self.frame_count:
            raise RuntimeError( ".yuv file does not seem to contain an integer number of frames" )

        self.frame_count = int(self.frame_count)

        self.mm = np.memmap( file_name, self.dtype, mode="r")

    def get_frame_count(self):
        return int(self.frame_count)
    
    def get_frame_yuv( self, frame_index ):

        if frame_index<0 or frame_index>=self.frame_count:
            raise RuntimeError( "The frame index is outside the range of available frames")

        offset = int(frame_index*self.frame_pixels)
        Y = self.mm[offset:offset+self.y_pixels]
        u = self.mm[offset+self.y_pixels:offset+self.y_pixels+self.uv_pixels]
        v = self.mm[offset+self.y_pixels+self.uv_pixels:offset+self.y_pixels+2*self.uv_pixels]

        return (np.reshape(Y,self.y_shape,'C'),np.reshape(u,self.uv_shape,'C'),np.reshape(v,self.uv_shape,'C'))

    def get_frame_rgb( self, frame_index ):

        (Y,u,v) = self.get_frame_yuv(frame_index)


        YUV = fixed2float( np.concatenate( (Y[:,:,np.newaxis],\
            convert420to444(u)[:,:,np.newaxis],\
            convert420to444(v)[:,:,np.newaxis]), axis=2), self.bit_depth)

#        RGB = (np.reshape( YUV, (self.y_pixels, 3), order='F' ) @ rgb_2020.transpose()).reshape( (*self.y_shape, 3 ), order='F' )
        
        RGB_709= (np.reshape( YUV, (self.y_pixels, 3), order='F' ) @ ycbcr2rgb.transpose()).reshape( (*self.y_shape, 3 ), order='F' )
        XYZ= (np.reshape( RGB_709, (self.y_pixels, 3), order='F' ) @ rgb709_xyz.transpose()).reshape( (*self.y_shape, 3 ), order='F' )
        RGB_2020= (np.reshape( XYZ, (self.y_pixels, 3), order='F' ) @ xyz2rgb2020.transpose()).reshape( (*self.y_shape, 3 ), order='F' )
        RGB= RGB_2020         
        
#        does the same thing as the above commented line
        return RGB

    def get_frame_rgb_rec2020( self, frame_index ):

        (Y,u,v) = self.get_frame_yuv(frame_index)


        YUV = fixed2float( np.concatenate( (Y[:,:,np.newaxis],\
            convert420to444(u)[:,:,np.newaxis],\
            convert420to444(v)[:,:,np.newaxis]), axis=2), self.bit_depth)

        RGB = (np.reshape( YUV, (self.y_pixels, 3), order='F' ) @ ycbcr2rgb_rec2020.transpose()).reshape( (*self.y_shape, 3 ), order='F' )

        return RGB
    
    

# vid_dir = "/local/bigscratch/ML4TMO/video_material"
## vid_name = "ghost_in_the_shell"
#vid_dir = "../../video"
#vid_name = "planet_earth_e2_mountains"
####
#fname = os.path.join( vid_dir, vid_name + "_4k", vid_name + "_960x540_420_2020_10b.yuv" )
#yf_hdr = YUVFile( fname )
####
#fname = os.path.join( vid_dir, vid_name + "_hd", vid_name + "_960x540_420_709_8b.yuv" )
#yf_sdr = YUVFile( fname )
####
##### #(Y,u,v) = yuvfile.get_frame_yuv(10)
#
#RGB_hdr = yf_hdr.get_frame_rgb_rec2020(120)
#RGB_sdr = yf_sdr.get_frame_rgb_rec2020(120)
##    
#pfs.view( RGB_hdr )
#pfs.view( RGB_sdr )
    
    
#for j in range(200):
#    i= j#+10000
#    RGB_hdr = yf_hdr.get_frame_rgb_rec2020(i)
#    RGB_sdr = yf_sdr.get_frame_rgb_rec2020(i)
##    
###    pfs.view( RGB_hdr )
###    pfs.view( RGB_sdr )
#    plt.imsave('Sample/HDR_2020/'+str(i)+'.png', np.clip(RGB_hdr,0,1))
#    plt.imsave('Sample/SDR_2020/'+str(i)+'.png', np.clip(RGB_sdr,0,1))
 
