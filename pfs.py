# Class for viewing numpy arrays with pfsview
# This was tested on Windows, Linux will be fixed later. 
#
# On Windows:
# You need to put the folder with the precompiled pfstools from
# https://sourceforge.net/projects/pfstools/files/pfstools_visual_studio_incomplete/
# in c:\Program Files (x86)\
# Installing the rest of pfstools in Cygwin is not needed to use pfsview. 

import os
import sys
import numpy as np
import struct
import shlex, subprocess
import platform

def which(program):
    import os
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None

class pfs:

    @staticmethod
    def cs_rgb2xyz( img ):
        M_rgb2xyz=np.array([ [0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193,0.1192, 0.9505] ])
        pix_cnt = img.shape[0]*img.shape[1]
        return (img.reshape( (pix_cnt, 3), order='F' ) @ M_rgb2xyz.transpose()).reshape( img.shape, order='F')

    @staticmethod
    def write_pfs( img, fh ):

        bin_fmt='{0}f'.format(img.size)

        if len(img.shape)==3 and img.shape[2]==3: # Colour image
            img_pfs = np.transpose(pfs.cs_rgb2xyz( img ),(2,0,1))
            bin=struct.pack(bin_fmt, *(img_pfs.flatten('C').astype('float')))
            pfs_header='PFS1\n{width} {height}\n3\n0\nX\n0\nY\n0\nZ\n0\nENDH'.format(width=img.shape[1], height=img.shape[0])
        else:   # Luminance image
            bin=struct.pack(bin_fmt, *(img.flatten('C').astype('float')))
            pfs_header='PFS1\n{width} {height}\n1\n0\nY\n0\nENDH'.format(width=img.shape[1], height=img.shape[0])

        fh.write(pfs_header.encode('ascii'))
        fh.write(bin)


    @staticmethod
    def view( img ):
        if platform.system()=="Linux":
            command_line = "pfsview"
            if not which(command_line):
                raise SystemExit( "pfsview not found. Check the comments in pfs.py for the instruction how to install it.")            
        else:
            command_line = "c:\\Program Files (x86)\\pfstools\\bin\\pfsview.exe"
            if not os.path.isfile(command_line):
                raise SystemExit( "pfsview not found. Check the comments in pfs.py for the instruction how to install it.")
        proc = subprocess.Popen(command_line, stdin=subprocess.PIPE)
        pfs.write_pfs(img, proc.stdin)


        
