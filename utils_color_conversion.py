#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 21:21:38 2021

@author: am2806
"""
#This file is just a copied python version of the matlab code in matlabtoolboxes
import numpy as np
#        RGB_709= (np.reshape( YUV, (self.y_pixels, 3), order='F' ) @ ycbcr2rgb.transpose()).reshape( (*self.y_shape, 3 ), order='F' )
#        XYZ= (np.reshape( RGB_709, (self.y_pixels, 3), order='F' ) @ rgb709_xyz.transpose()).reshape( (*self.y_shape, 3 ), order='F' )
#        RGB_2020= (np.reshape( XYZ, (self.y_pixels, 3), order='F' ) @ xyz2rgb2020.transpose()).reshape( (*self.y_shape, 3 ), order='F' )
#        RGB= RGB_2020 
      
        
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

        
def rgb2xyz_m2020(inp):

        
    XYZ = (np.reshape( inp, (540*960, 3), order='F') @ rgb2020_xyz.transpose()).reshape( (540,960, 3 ), order='F')
    return XYZ


def xyz2rgb_709(xyz):
    
    rgb709_xyz_inverse = np.linalg.inv(np.array([[0.4124, 0.3576, 0.1805],
                      [0.2126, 0.7152, 0.0722],
                      [0.0193, 0.1192, 0.9505]], dtype=np.float32)  ) 

    rgb_709= (np.reshape( xyz, (540*960, 3), order='F' )@rgb709_xyz_inverse.transpose()).reshape( (540,960, 3 ), order='F' )
    
    return rgb_709       