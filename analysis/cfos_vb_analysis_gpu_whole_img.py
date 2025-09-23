
import tifffile
import numpy as np
import os
import cv2
import json
import time
import math
import pandas as pd
# import SimpleITK as sitk
import pickle

# import seaborn as sns
from scipy import stats
import gc
from tifffile import TiffFile
from joblib import Parallel, delayed
# from skimage import io
import tifffile
from scipy.ndimage import zoom
import traceback
import gzip
import shutil
# from numba import jit

import sys
sys.path.append("/mnt/gpu_data/data1/kinoshita/")
import cfospy

import matplotlib.pyplot as plt

from multiprocessing import Pool
plt.gray()

import json, os.path, os, re, time
# import joblib
import subprocess
import nibabel as nib
import scipy.spatial
from scipy.signal import convolve
from matplotlib.colors import Normalize, LinearSegmentedColormap, BoundaryNorm, ListedColormap, hsv_to_rgb
import cfospy

#env CUDA_VISIBLE_DEVICES=0 python3 cfos_vb_analysis_gpu_whole_img.py

r=100
# rIDs =[286]   #ABPV#[ 88, 795, 390, 242,  351,  186, 525]  #[1079, 1088] #MGv, MGm ##[262, 149, 194,  872,  147, 557 ]
vx = 20
vb_r=8
mo=1
size=5   #size*size slice
s=0 #offset image
# op="fdr"  #"None"
ncore = 20
ants_dir_name = "ANTsR50"
vb_pre_file = "whole_vb_a_new"  #"region_vb_ci"
calc_type = "count"  #
conut_file = "cell_table_combine_I_ai_fpr0.5.npy"

op_alpha = True
n=3
b=0.01

blockdim_x =  1

src = "/mnt/gpu_data/data1/kinoshita/"
dst = "/mnt/gpu_data/data8/kinoshita_cfos/"

rdir = src + "CUBIC_R_atlas_ver5/"
cfos_fol=src + "yamashitaData1/230828circadian_Data1/230828circadian_1st_Reconst/"
exp = "1st"
savedir = dst + "cfos_app/"

class mapping_to_atlas():
    def __init__(self,data_atlas_path, ants_dir_name, data_ants_path, before_ants_file):
        self.ants_voxel_unit = Resize_um
        self.original_voxel_unit = Resize_um
        self.img_voxel_unit = Resize_um

        
        self.atlas_tif_path = data_atlas_path
        self.atlas_nii_path = self.atlas_tif_path.replace(".tif",".nii.gz")
        
        #for image size check
        
        
#         self.flip_tif_path =  os.path.join(data_parent_path,"{}/{}/ANTs/density_img_50um_flip.tif".format(organ_name,sample_name))
#         self.flip_txt_path = os.path.join(data_parent_path,"{}/{}/ANTs/flip_memo.txt".format(organ_name,sample_name))

        self.ants_dst_dir = os.path.join(data_ants_path, ants_dir_name+"/")
    
        self.sample_resize_img_path =  os.path.join(data_ants_path,before_ants_file)
        
        #self.flip_tif_path --->  self.sample_nii_path
        self.sample_nii_path = os.path.join(data_ants_path ,before_ants_file.replace(".tif",".nii.gz"))
        
        self.before_ants_np_path = os.path.join(data_ants_path,"all_points_um.npy")
        
        if not os.path.exists(self.ants_dst_dir):
            print("make ants folder {}".format(self.ants_dst_dir))
            os.makedirs(self.ants_dst_dir)
        else:
            print("{} already exists".format(self.ants_dst_dir))

        self.moving_nii_path = self.sample_nii_path
        
        self.output_nii_path = os.path.join(self.ants_dst_dir, "after_ants.nii.gz")
        
        self.prefix_ants ="/opt/ANTs/bin/"# "/usr/local/ANTs/"
        
#         self.moving_points_path = os.path.join(self.ants_dst_dir,"moving.csv")   #?
#         self.moved_points_path = os.path.join(self.ants_dst_dir,"moved.csv")#?
#         self.moved_denstity_path = os.path.join(self.ants_dst_dir,"moved_denstity.tif")
    
        
    def tif2nii(self, tif_path, nii_path, nii_voxel_unit):
        tif_img = tifffile.imread(tif_path)
#         nii_img = nib.Nifti1Image(np.swapaxes(tif_img,0,2), affine=None)
        nii_img = nib.Nifti1Image(tif_img, affine=None)
        aff = np.diag([-nii_voxel_unit,-nii_voxel_unit,nii_voxel_unit,1])
        nii_img.header.set_qform(aff, code=2)
        nii_img.to_filename(nii_path)
        return



    def run_antsRegistration(self,prefix_ants, atlas_file, moving_file, dst_dir, threads):
        cmd = "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={THREADS} && "
        cmd += "{EXECUTABLE} -d 3 "
        cmd += "--initial-moving-transform [{ATLAS_FILE},{MOVING_FILE},1] "
        cmd += "--interpolation Linear "
        cmd += "--use-histogram-matching 0 "
        cmd += "--winsorize-image-intensities [0.05,1.0] "  
        cmd += "--float 0 "
        cmd += "--output [{DST_PREFIX},{WARPED_FILE},{INVWARPED_FILE}] "
        cmd += "--transform Affine[0.1] --metric MI[{ATLAS_FILE},{MOVING_FILE},1,128,Regular,0.5] --convergence [10000x10000x10000,1e-5,15] --shrink-factors 4x2x1 --smoothing-sigmas 2x1x0vox "
        cmd += "--transform SyN[0.1,3.0,0.0] --metric CC[{ATLAS_FILE},{MOVING_FILE},1,5] --convergence [300x100x30,1e-6,10] --shrink-factors 4x2x1 --smoothing-sigmas 2x1x0vox"

#         cmd += "--transform Affine[0.1] --metric MI[{ATLAS_FILE},{MOVING_FILE},1,128,Regular,0.5] --convergence [10000x10000x10000,1e-5,15] --shrink-factors 8x4x2 --smoothing-sigmas 3x2x1vox "
#         cmd += "--transform SyN[0.1,3.0,0.0] --metric CC[{ATLAS_FILE},{MOVING_FILE},1,4] --convergence [500x500x500x50,1e-6,10] --shrink-factors 8x4x2x1 --smoothing-sigmas 3x2x1x0vox"
        cmd = cmd.format(
            THREADS = threads,
            EXECUTABLE = os.path.join(prefix_ants, "antsRegistration"),
            DST_PREFIX = os.path.join(dst_dir, "F2M_"),
            WARPED_FILE = os.path.join(dst_dir, "F2M_Warped.nii.gz"),
            INVWARPED_FILE = os.path.join(dst_dir, "F2M_InvWarped.nii.gz"),
            ATLAS_FILE = atlas_file,
            MOVING_FILE = moving_file,
        )
        print("[*] Executing : {}".format(cmd))
        sp.call(cmd, shell=True)
        return


    #def set_
    
    #scale of list is atlas_voxel_unit
    def make_density_image(self,list_x,list_y,list_z,size_x,size_y,size_z):
        depth,height,width = size_z,size_y,size_x
        density_img,_ = np.histogramdd(
            np.vstack([list_z,list_y,list_x]).T,
            bins=(depth, height, width),
            range=[(0,depth-1),(0,height-1),(0,width-1)]
        )
        return density_img
    
    def run_antsApplyTransformsToPoints(self, prefix_ants, src_csv, dst_csv, ANTs_image_dir):
        cmd = "{EXECUTABLE} "
        cmd += "-d 3 "
        cmd += "-i {SRC_CSV} "
        cmd += "-o {DST_CSV} "
        cmd += "-t [{AFFINE_MAT},1] "
        cmd += "-t {INVWARP_NII}"
        cmd = cmd.format(
            EXECUTABLE = os.path.join(prefix_ants, "antsApplyTransformsToPoints"),
            AFFINE_MAT = os.path.join(ANTs_image_dir, "F2M_0GenericAffine.mat"),
            INVWARP_NII = os.path.join(ANTs_image_dir, "F2M_1InverseWarp.nii.gz"),
            SRC_CSV = src_csv,
            DST_CSV = dst_csv,
        )
        #print("[*] Executing : {}".format(cmd))
        # supress output
        with open(os.devnull, 'w') as devnull:
            sp.check_call(cmd, shell=True, stdout=devnull)
        return
    
    def run_antsApplyTransforms(self, prefix_ants, src_file, atlas_file, dst_file, ANTs_image_dir):
        cmd = "{EXECUTABLE} "
        cmd += "-d 3 "
        cmd += "-e 0 "  #choose 0/1/2/3 mapping to scalar/vector/tensor/time-series
        cmd += "-i {SRC_FILE} "
        cmd += "-r {REF} "
        cmd += "-o {DST_FILE} "
        cmd += "-n {INTERP} "
        cmd += "-t {INVWARP_NII} "
        cmd += "-t {AFFINE_MAT} "
#         cmd += "-t [{AFFINE_MAT},1] "
#         cmd += "-t {INVWARP_NII} "
       
        cmd = cmd.format(
            EXECUTABLE = os.path.join(prefix_ants, "antsApplyTransforms"),
            SRC_FILE = src_file,
            REF = atlas_file,
            DST_FILE = dst_file,
            INTERP = "Linear",
            INVWARP_NII = os.path.join(ANTs_image_dir, "F2M_1Warp.nii.gz"),
            AFFINE_MAT = os.path.join(ANTs_image_dir, "F2M_0GenericAffine.mat"),
        
        )
        #print("[*] Executing : {}".format(cmd))
        # supress output
        with open(os.devnull, 'w') as devnull:
            sp.check_call(cmd, shell=True, stdout=devnull)
        return 
    
#2D slice tif images -> 3D tif image
def stack_2d_images(path):
    
    imgs = os.listdir(path)

    img0 = tifffile.imread(path + "/"+imgs[0])
    
    width = img0.shape[0]
    height  = img0.shape[1]
    
    stack_image = np.zeros((width, height, len(imgs)))
    
    for i, img_f in enumerate(imgs):
        if "ANTs" in img_f or "ANTsR" in img_f:
            continue
        img = tifffile.imread(path + "/"+img_f)
        stack_image[:,:,i]=img
    return stack_image



def convert_nii_to_tiff(nii_file, output_tiff_file):
    # NIfTI ファイルを読み込む
    nii_data = nib.load(nii_file)
    # NIfTI データを NumPy 配列に変換
    nii_array = nii_data.get_fdata()
    # NumPy 配列を TIFF ファイルに保存
    tifffile.imwrite(output_tiff_file, nii_array)

def decompress_gzip(input_file, output_file):
    with gzip.open(input_file, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def convolve_3d_array(data, kernel):
    # 各チャンネルに対して畳み込みを実行
    result = convolve(data, kernel, mode='same',method='direct')

    return result
    
from statsmodels.stats.multitest import multipletests

# calculate max correlation
def calc_max_corr(vec, norm_base_s, norm_base_c):
    vec_m = vec - np.mean(vec, axis = 1).reshape(-1,1)
    norm_vec = (vec_m)/np.sqrt(np.sum((vec_m)*(vec_m), axis=1)).reshape(-1,1)
    # print("norm_base_c", norm_base_c.shape)
    # print("norm_base_s", norm_base_s.shape)
    # print("norm_vec", norm_vec.shape)
    # print("np.dot(norm_base_c, norm_vec.T)", np.dot(norm_base_c, norm_vec.T))
    # print("np.dot(norm_base_s, norm_vec.T)", np.dot(norm_base_s, norm_vec.T))
    max_corr_value = np.sqrt(np.power(np.dot(norm_base_c, norm_vec.T), 2) + np.power(np.dot(norm_base_s, norm_vec.T), 2))
    # print(max_corr_value)
    max_corr_phase = np.arctan2(np.dot(norm_base_s, norm_vec.T), np.dot(norm_base_c, norm_vec.T))
    return max_corr_value, max_corr_phase


def norm_bases(n_dp, n_dp_in_per):
    base_s = np.sin(np.arange(n_dp)/n_dp_in_per*2*np.pi)
    base_c = np.cos(np.arange(n_dp)/n_dp_in_per*2*np.pi)
    base_s_m = base_s - np.mean(base_s)
    base_c_m = base_c - np.mean(base_c)
    norm_base_s = (base_s_m)/np.sqrt(np.sum((base_s_m)*(base_s_m)))
    norm_base_c = (base_c_m)/np.sqrt(np.sum((base_c_m)*(base_c_m)))
    return norm_base_s, norm_base_c


def max_corr_pval(num_datapoints, max_corr):
    """ pvalue of the max Pearson correlations 
        max corr: the value of max correlation
        return: cumulative density
    """
    n = num_datapoints - 3
    p = np.power((1 - np.power(max_corr, 2)), n / 2)
    return p


def sem_ratio(avg_vector, sem_vector):
    """ ratio of the avg_vecotr at which the vector reaches the SEM sphere
        avg_vector: average vector
        sem_vector: SEM vector
        return: the ratio of the average vector from the origin to the SEM sphere over the average vector
    """
   
    ratio = 1.0 - 1.96 / np.sqrt(np.sum(np.power(avg_vector / sem_vector, 2))) # 1.96 is the 95% confidence interval
    return max(0, ratio)


def costest(avg_vec, sem_vec, n_dp, n_dp_in_per):
    # prepare bases
    norm_base_s, norm_base_c = norm_bases(n_dp, n_dp_in_per)

    ## prepare vectors to handle nan
    avg_vector = avg_vec.copy()
    
    sem_vector = sem_vec.copy()
    # each SEM needs to be >0
    sem_vector[~(sem_vector>0)] = np.nanmax(sem_vector)
    # replace nan in SEM with an effectively infinite SEM
    sem_vector[np.isnan(avg_vector)] = np.nanmax(sem_vector)*1000000
    # replace nan in AVG with median
    avg_vector[np.isnan(avg_vector)] = np.nanmedian(avg_vector)
    # taking the SEM into account
    sem_r = sem_ratio(avg_vector, sem_vector)

    # tuple of max-correlation value and the phase of max correlation
    mc, mc_ph = calc_max_corr(avg_vector, norm_base_s, norm_base_c)

    adj_mc = mc * sem_r
    p_org = max_corr_pval(n_dp, mc)
    p_sem_adj = max_corr_pval(n_dp, adj_mc)
    
    # max-correlation value, phase of max correlation, original p, SEM adjusted p
    return mc, mc_ph, p_org, p_sem_adj

import math
def rad2ph(rad):
#     return (int((2*np.pi+rad)*180/np.pi*24/360),  int((rad)*180/np.pi*24/360))[rad>=0]
    if math.isnan(rad):
        return np.nan
    else:
        return (round((2*np.pi+rad)*180/np.pi*24/360, 1),  round((rad)*180/np.pi*24/360, 1))[rad>=0]
    
# cfos_fol="/mnt/data1/yamashitaData1/231012_circadian_2nd_Data1/231012_circadian_2nd_Reconst/"
# exp = "2nd"
# cfos_fol="/home/gpu_data/data1/yamashitaData1/230828circadian_Data1/230828circadian_1st_Reconst/"
# exp = "1st"
# savedir = "/home/gpu_data/data8/cfos_app/"

CT_li = np.arange(0,48,4)
sample_ids = np.arange(1,7,1)

reconsts = os.listdir(cfos_fol)
sample_names=[]
colors=[]
data_parent_paths=[]
data_moving_paths =[]

for CT in CT_li:
#     if CT<44:
#         continue
    for sample_id in sample_ids:
#         if sample_id >1:
#             continue
        sample = "CT"+str(CT)+"_"+str(sample_id).zfill(2)

        for reconst in reconsts:
#             sample_names.append(("_").join(reconst.split("_")[1:3]))

            if sample in reconst:
                sample_names.append(sample)
                for color in os.listdir(cfos_fol+reconst):
                    if "cfos" in color:
                        data_moving_paths.append(cfos_fol+reconst+"/"+color)
                    else:
                        data_parent_paths.append(cfos_fol+reconst+"/"+color)

print(len(data_parent_paths))   
print(len(data_moving_paths))
print(sample_names)


class calc_vb_phase:
    def __init__(self, rdir, savedir, vb_pre_file,  mask0, vx, r, vb_r, mo):
        self.region = "whole"#ca.df_allen[ca.df_allen["ID"]==rID]["acronym"].iloc[0]
        self.vx = vx
        self.r = r
        self.savedir = savedir
        self.vb_r = vb_r
        self.mo = mo
        self.rdir = rdir
        
        region = "whole"
        print(self.region)
        # print(rID) region = ca.df_allen[ca.df_allen["ID"]==rID]["acronym"].iloc[0]
        
       
        # print(region)
        # print("rID", rID)
                
        if "/" in region:
            region = region.replace("/", "_")

      
        # mask0 = ca.get_atlas_mask()
        self.total_b_num = len(np.where(np.ravel(mask0)==1)[0])

        v_ind = np.where(mask0)
        nv_ind = np.where(~mask0)
        # v_ind_xl = v_ind[2][np.where(v_ind[2]<xc)[0]]
        #         print(len(v_ind_xl))
        # v_ind_xr = v_ind[2][np.where(v_ind[2]>=xc)[0]]
        #         print(len(v_ind_xr))
        # xmin_l =  np.min(v_ind_xl)
        xmin = np.min(v_ind[2])
        ymin = np.min(v_ind[1])
        zmin = np.min(v_ind[0])
        # xmax_l = np.max(v_ind_xl)
        xmax = np.max(v_ind[2])
        ymax = np.max(v_ind[1])
        zmax = np.max(v_ind[0])

        
        xmin= int(xmin -(xmax-xmin)/r)
        ymin = int(ymin -(ymax-ymin)/r)
        zmin =int(zmin -(zmax-zmin)/r)
        xmax = int(xmax +(xmax-xmin)/r)
        ymax = int(ymax +(ymax-ymin)/r)
        zmax = int(zmax +(zmax-zmin)/r)
            
        if xmin < 0 :
            xmin =0
        # if xmin_l>xc:
        #     xmin_l = xc
        if xmax > ca.x_num:
            xmax = ca.x_num
        # if xmin_r<xc:
        #     xmin_r = xc
        if ymin < 0 :
            ymin =0
        if ymax > ca.y_num:
            ymax = ca.y_num
        if zmin < 0 :
            zmin =0
        if zmax > ca.z_num:
            zmax = ca.z_num
        

            
        self.xmin = xmin 
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax
   
        
        vol = (xmax-xmin)*(ymax-ymin)*(zmax-zmin)
        print("voxel volume", vol)
        print("moving", self.mo)
        
        self.vb_pre_file = vb_pre_file
        
        self.root_fol_count= vb_pre_file + "/{}um/{}/vb{}_mo{}/".format(self.vx, self.region, self.vb_r, self.mo)


#         return xmin, xmax, ymin, ymax, zmin, zmax
        # if not "whole"in vb_pre_file:
            # if calc_type =="count_ratio":
            #     calc_fol = "region_vb"#cell count
            # elif calc_type =="count":
            #     calc_fol = "region_vb_a"#
            # elif calc_type =="cell_intensity_ratio":
            #     calc_fol = "region_vb_cir"#
            # elif calc_type =="cell_intensity":
            #     calc_fol = "region_vb_ci"#
        # else:
        if calc_type =="count_ratio":
            calc_fol = "whole_vb_new"#cell count
        elif calc_type =="count":
            calc_fol = "whole_vb_a_new"#
        elif calc_type =="cell_intensity_ratio":
            calc_fol = "whole_vb_cir"#
        elif calc_type =="cell_intensity":
            calc_fol = "whole_vb_ci"#
                
        self.calc_fol = calc_fol
            
        self.fol =  "/"+calc_fol+"/{}um/{}/vb{}_mo{}/".format(self.vx, self.region, self.vb_r, self.mo)

        self.dst = self.savedir+self.fol+"/"
        os.makedirs(self.dst, exist_ok=True)
        print(self.dst)
    
    
    
    def get_vb_ID(self):
        img_vx_ID = np.arange(0, len(ca.voxel_ID_order_all)).reshape(ca.x_num, ca.y_num, ca.z_num)
        vb_ID = {}
        c = 0
#         if not os.path.exists(savedir+"/region_vb{}_mo{}_r{}/{}/".format(self.vb_r, self.mo, self.r, self.region)+"vb_ID.pkl"):
        
        self.x_b_num = int((self.xmax-self.xmin)/self.mo+1)
        self.y_b_num = int((self.ymax-self.ymin)/self.mo+1)
        self.z_b_num = int((self.zmax-self.zmin)/self.mo+1)
        self.total_b_num = self.x_b_num*self.y_b_num*self.z_b_num
        
        if not os.path.exists(self.dst+"vb_ID.pkl"):
        
            for x in range(self.x_b_num):
                for y in range(self.y_b_num):
                    for z in range(self.z_b_num):
                        if self.mo !=1:
                        
                            if self.xmin+x*self.mo - self.mo +1 <0:
                                xmin2 = 0
                            else:
                                xmin2 =  self.xmin+x*self.mo - self.mo +1

                            if self.xmin+x*self.mo +  self.mo -1 > ca.x_num:
                                xmax2 = ca.x_num
                            else:
                                xmax2 = self.xmin+x*self.mo +  self.mo -1

                            if self.ymin+y*self.mo - self.mo +1 <  0:
                                ymin2=0
                            else:
                                ymin2 = self.ymin+y*self.mo- self.mo +1

                            if self.ymin+y*self.mo +  self.mo -1 >ca.y_num:
                                ymax2=ca.y_num
                            else:
                                ymax2 = self.ymin+y*self.mo+  self.mo -1

                            if self.zmin+z*self.mo- self.mo +1 < 0:
                                zmin2 = 0
                            else:
                                zmin2 = self.zmin+z*self.mo- self.mo +1

                            if self.zmin+z*self.mo +  self.mo -1 > ca.z_num:
                                zmax2 = ca.z_num
                            else:
                                zmax2 = self.zmin+z*self.mo +  self.mo -1
                                
                            vb_ID[c] = np.ravel(img_vx_ID[xmin2:xmax2, ymin2:ymax2, zmin2:zmax2])
                    

                        else:
                            if self.xmin+x*self.mo>=ca.x_num or self.ymin+y*self.mo>=ca.y_num or self.zmin+z*self.mo>=ca.z_num:
                                continue

                            vb_ID[c] = img_vx_ID[self.xmin+x*self.mo, self.ymin+y*self.mo, self.zmin+z*self.mo]
                        c+=1

            with open(self.dst+"vb_ID.pkl", "wb") as tf:
                pickle.dump(vb_ID, tf)
#         return vb_ID

        
    def make_cos_vb(self, sample_names):
        
        if not os.path.exists(self.dst + "/cos_1st2nd.csv"):
            CT_li = np.arange(0, 48, 4)
            CT_li2 = np.arange(0, 96, 4)
            sample_ids = np.arange(1, 7, 1)
            # for l, exp in enumerate(exps):
            
            
            vb_exp = np.zeros((self.total_b_num, len(CT_li)*len(sample_ids)), dtype = "float32")
            
            # print("vb_exp", vb_exp.shape)
            exp = "1st"
            os.makedirs(self.savedir+exp+ "/"+self.fol, exist_ok=True)
            
            if not os.path.exists(self.savedir+exp+"/"+self.fol + "vb_CT_{}".format(exp)):
                for m, CT in enumerate(CT_li):
                    for n,sample_id in enumerate(sample_ids):
                        sample = "CT"+str(CT)+"_"+str(sample_id).zfill(2)
                        print(sample)
                        cf =self.savedir+exp+ "/"+self.root_fol_count+ "/"+sample +"_vb_CT_{}.bin".format(exp)
                        rectype = np.dtype(np.int32)
                        vb_bin = np.fromfile(cf, dtype=rectype)#.astype("float32")
                        print("vb_bin", vb_bin.shape)
                        vb_exp[:,m*len(sample_ids)+n] = vb_bin
                if calc_type == "count_ratio":
                    T_cells = np.load(self.savedir+exp+"/total_cell_nums.npy")
                    vb_exp = vb_exp/T_cells.reshape(-1, len(CT_li)*len(sample_ids))  #ratio
                elif calc_type == "cell_intensity_ratio":
                    T_cells = np.load(self.savedir+exp+"/total_cell_intenses.npy")
                    vb_exp = vb_exp/T_cells.reshape(-1, len(CT_li)*len(sample_ids))  #ratio
               
                elif calc_type == "count":
                    vb_exp = vb_exp
                elif calc_type == "cell_intensity":
                    vb_exp = vb_exp
                
                np.save(self.savedir+exp+"/"+self.fol + "vb_CT_{}".format(exp), vb_exp)
            
            
            
            exp = "2nd"
            os.makedirs(self.savedir+exp+ "/"+self.fol, exist_ok=True)
            if not os.path.exists(self.savedir+exp+"/"+self.fol + "vb_CT_{}".format(exp)):
                for m, CT in enumerate(CT_li):
                    for n,sample_id in enumerate(sample_ids):
                        sample = "CT"+str(CT)+"_"+str(sample_id).zfill(2)
                        print(sample)
                        cf =self.savedir+exp+ "/"+self.root_fol_count + "/"+sample +"_vb_CT_{}.bin".format(exp)
                        rectype = np.dtype(np.int32)
                        vb_bin = np.fromfile(cf, dtype=rectype)#.astype("float32")
                        print("vb_bin", vb_bin.shape)
                        vb_exp[:,m*len(sample_ids)+n] = vb_bin
                        
                if calc_type == "count_ratio":
                    T_cells = np.load(self.savedir+exp+"/total_cell_nums.npy")
                    vb_exp = vb_exp/T_cells.reshape(-1, len(CT_li)*len(sample_ids))  #ratio
                elif calc_type == "cell_intensity_ratio":
                    T_cells = np.load(self.savedir+exp+"/total_cell_intenses.npy")
                    vb_exp = vb_exp/T_cells.reshape(-1, len(CT_li)*len(sample_ids))  #ratio
               
                elif calc_type == "count":
                    vb_exp = vb_exp
                elif calc_type == "cell_intensity":
                    vb_exp = vb_exp
                
                np.save(self.savedir+exp+"/"+self.fol + "vb_CT_{}".format(exp), vb_exp)
            
            
            
            
            
            df_all=[]
            df_all_r=[]
            cols =[]
            
            for m, CT in enumerate(CT_li2):
                for n,sample_id in enumerate(sample_ids):
                    sample = "CT"+str(CT)+"_"+str(sample_id).zfill(2)
                    cols.append(sample)
                    
            
            path1 = self.savedir+"1st/"+self.fol + "vb_CT_{}.npy".format("1st")
            CT_np1 = np.load(path1)
                
                
                
            path2 = self.savedir+"2nd/"+self.fol+ "vb_CT_{}.npy".format("2nd")
            CT_np2 = np.load(path2)
            

            df = np.hstack([CT_np1, CT_np2])
            del CT_np1
            del CT_np2
            gc.collect()

            df = pd.DataFrame(df)
            df.columns=cols
            df.insert(0, "id", np.arange(0, self.total_b_num))

            CT_mean_li = []
            CT_se_li = []
            for i, CT in enumerate(CT_li2):
                df_mean = df.iloc[:, 1+i*len(sample_ids):1+(i+1)*len(sample_ids)].mean(axis=1)
                df_sem = df.iloc[:, 1+i*len(sample_ids):1+(i+1)*len(sample_ids)].std(axis=1)/np.sqrt(len(sample_ids))

                CT_mean_li.append(df_mean)
                CT_se_li.append(df_sem)

            df_r_mean = pd.concat(CT_mean_li, axis=1)
            df_r_se = pd.concat(CT_se_li, axis=1)

            # print(df_r_mean)


            #periodicity by cell count
            n_dp = 24 #np.ones(len(df_mean))*12
            n_dp_per = 6#np.ones(len(df_mean))*6
            # mc, mc_ph, p_org, p_sem_adj = costest(df_m_i[0], df_se_i[0], 12, 6)
            mc, mc_ph, p_org, p_sem_adj = costest(np.array(df_r_mean), np.array(df_r_se), n_dp, n_dp_per)

            nan_id = np.isnan(p_sem_adj)
            nonan_id = np.where(nan_id==False)[0]

            phase_li = list(map(rad2ph, mc_ph))
            # acronyms= [ca.df_allen[ca.df_allen["ID"]==rID]["acronym"].iloc[0] for rID in df["id"]]
            # node_names= [ca.df_allen[ca.df_allen["ID"]==rID]["node_name"].iloc[0] for rID in df["id"]]
            per_li = np.ones(len(df))*24


            cos_v_df = pd.DataFrame({"id":df["id"],  "ADJ.P":p_sem_adj,  "PER":per_li, "LAG":phase_li, "RAD":mc_ph, "max_corr":mc})

            p_bh_np = np.ones(len(p_sem_adj))
            bool_, p_bh_li, _, _ =  multipletests(p_sem_adj[nonan_id],  method="fdr_bh")

            p_bh_np[nonan_id] = p_bh_li

            cos_v_df.insert(3, "BH.Q", p_bh_np)


            # print(cos_v_df)
            cos_v_df  =  pd.concat([cos_v_df, df.iloc[:, 1:]],axis=1)

            # cos_v_df = cos_v_df.sort_values("BH.Q")

            cos_v_df.to_csv(self.dst + "cos_1st2nd.csv")
        
#         return cos_v_df
    
        
    def make_vb_image(self, op="None", op_alpha = True, n=4, b=0.2):
#         op ="fdr"
        # op="None"
        a = (1-b)/(-1)**n
        

        with open(self.savedir+self.fol+ "vb_ID.pkl", mode="rb") as f:
            vb_ID = pickle.load(f)
        cos_v_df = pd.read_csv(self.savedir+self.fol + "cos_1st2nd.csv", index_col=0)

        # vb_ID = pickle.(savedir+"/region_vb{}_mo{}/{}/".format(vb_r, mo, region)+ "vb_ID.pkl")
        img_vx = np.ones(len(ca.voxel_ID_order_all))*(-1)  #解析したボクセルブロック以外のボクセルは-1　→ black

        if op =="fdr":
            fdr_li_all = np.zeros(len(ca.voxel_ID_order_all))
            fdr_li = cos_v_df["BH.Q"] < 0.1
            if op_alpha == True:
                alpha_li_all = np.zeros(len(ca.voxel_ID_order_all))
                fdr_vs = np.array(cos_v_df["BH.Q"])
                alpha_li = -np.log10(fdr_vs)
                a_max = np.max(alpha_li)
                alpha_li = alpha_li/a_max
                
                alpha_li = (-a*(alpha_li-1)**n + 1)*255
                print(np.max(alpha_li))
                print(np.min(alpha_li))

        ph_li = (cos_v_df["LAG"]/24)
        ph_li = 1-ph_li
        ph_li = [(hue + 1/3 - 1) if (hue + 1/3) > 1 else (hue + 1/3) for hue in ph_li]

        

        for i in range(len(cos_v_df)):
            v_IDs=vb_ID[i]
            img_vx[v_IDs] = ph_li[i]
            if op =="fdr":
                fdr_li_all[v_IDs] = fdr_li[i]
                if op_alpha == True:
                     alpha_li_all[v_IDs] = alpha_li[i]

        # hsv_li = np.array([hsv_to_rgb(i) for i in img_vx] if not math.isnan(i))
        hsv_li=[]
        for n, i in enumerate(img_vx):
            if op=="fdr" and op_alpha ==False:
                if not math.isnan(i) and not i==-1 and fdr_li_all[n]==1:
                    hsv_li.append(np.append(hsv_to_rgb([i, 1, 1])*255, 255))
            #         print(hsv_to_rgb([i, 1, 1]))
                elif math.isnan(i) or fdr_li_all[n]!=1:
        #             hsv_li.append((255,255,255, alpha))  #white  #解析したけど振動なし、phaseなし
                    hsv_li.append((0,0,0, 255))  #black  #解析したけど振動なし、phaseなし
                else:
                    hsv_li.append((0,0,0, 255)) #black
                    
            elif op=="fdr" and op_alpha ==True:
                if not math.isnan(i) and not i==-1 and fdr_li_all[n]==1:
                    hsv_li.append(np.append(hsv_to_rgb([i, 1, 1])*255, alpha_li_all[n]))
            #         print(hsv_to_rgb([i, 1, 1]))
                elif math.isnan(i) or fdr_li_all[n]!=1:
        #             hsv_li.append((255,255,255, alpha))  #white  #解析したけど振動なし、phaseなし
                    hsv_li.append((0,0,0,255))  #black  #解析したけど振動なし、phaseなし
                else:
                    hsv_li.append((0,0,0, 255)) #black
            

            else:
                if not math.isnan(i) and not i==-1:
                    hsv_li.append(np.append(hsv_to_rgb([i, 1, 1])*255, 255))
            #         print(hsv_to_rgb([i, 1, 1]))
                elif math.isnan(i):
        #             hsv_li.append((255,255,255, alpha))  #white  #解析したけど振動なし、phaseなし
                    hsv_li.append((0,0,0, 255))  #black  #解析したけど振動なし、phaseなし
                else:
                    hsv_li.append((0,0,0, 255)) #black

        print(np.array(hsv_li).shape)

        hsv_li = np.array(hsv_li).astype("uint8")

        img_vx = np.swapaxes(hsv_li.reshape(ca.x_num, ca.y_num, ca.z_num, 4), 0, 2)
        print(img_vx.shape)
        
        return img_vx
    
    def make_vb_image2(self, atlas_mask, op="None", op_alpha = True, n=4, b=0.2):
#         op ="fdr"
        # op="None"
#         n = 6
        a = (1-b)/(-1)**n
# y=b*(math.e)**(np.log(1/b)*a*x)
        if not os.path.exists(self.dst+"/vb_{}_img.tif".format( op)):

            # with open(self.savedir+self.fol+ "vb_ID.pkl", mode="rb") as f:
            #     vb_ID = pickle.load(f)
            cos_v_df = pd.read_csv(self.savedir+self.fol + "cos_1st2nd.csv", index_col=0)

            # vb_ID = pickle.(savedir+"/region_vb{}_mo{}/{}/".format(vb_r, mo, region)+ "vb_ID.pkl")
            img_vx = np.zeros((len(ca.voxel_ID_order_all),4),dtype="uint8") #解析したボクセルブロック以外のボクセルは-1　→ black
            
            
            if op =="fdr":
                # fdr_li_all = np.zeros(len(ca.voxel_ID_order_all))
                fdr_li = cos_v_df["BH.Q"] < 0.1
                
                if op_alpha == True:
                    # alpha_li_all = np.zeros(len(ca.voxel_ID_order_all))
                    fdr_vs = np.array(cos_v_df["BH.Q"])
                    alpha_li = -np.log10(fdr_vs)
                    a_max = np.max(alpha_li)
                    alpha_li = alpha_li/a_max
                    
                    alpha_li = (-a*(alpha_li-1)**n + 1)*255
                    print("alpha_max", np.max(alpha_li))
                    print("alpha_min",np.min(alpha_li))
    #                 alpha_li = np.clip(alpha_li, a_lim, 1)*255
                else:
                    alpha_li = np.ones(len(cos_v_df))*255
            else:
                alpha_li = np.ones(len(cos_v_df))*255
                    

            ph_li = (cos_v_df["LAG"]/24)
            ph_li = 1-ph_li
            ph_li = [(hue + 1/3 - 1) if (hue + 1/3) > 1 else (hue + 1/3) for hue in ph_li]

            brain_ind = np.where(np.ravel(np.swapaxes(atlas_mask, 0, 2))==1)[0]

            for i in range(len(cos_v_df)):
                vx_ind = brain_ind[i]
                if op =="fdr":
                    if fdr_li[i] == 1:
                        img_vx[vx_ind] = np.append(hsv_to_rgb([ph_li[i], 1, 1])*255, alpha_li[i])
                
                else:
                    img_vx[vx_ind] = np.append(hsv_to_rgb([ph_li[i], 1, 1])*255, alpha_li[i])
                

            
            # hsv_li=[]
            # for n, i in enumerate(img_vx):
            #     if op=="fdr" :
            #         if not math.isnan(i) and not i==-1 and fdr_li_all[n]==1:
            #             hsv_li.append(np.append(hsv_to_rgb([i, 1, 1])*255, alpha_li_all[n]))
                        
            #         elif math.isnan(i) or fdr_li_all[n]!=1:
            
            #             hsv_li.append((0,0,0, 0))  #black  #解析したけど振動なし、phaseなし
            #         else:
            #             hsv_li.append((0,0,0, 0)) #black

            #     else:
            #         if not math.isnan(i) and not i==-1:
            #             hsv_li.append(np.append(hsv_to_rgb([i, 1, 1])*255, 255))
            #     #       
            #         elif math.isnan(i):
            # #                                         
            #             hsv_li.append((0,0,0, alpha))  #black  #解析したけど振動なし、phaseなし
            #         else:
            #             hsv_li.append((0,0,0, alpha)) #black

            # print(np.array(hsv_li).shape)

            # hsv_li = np.array(hsv_li).astype("uint8")

            img_vx = np.swapaxes(img_vx.reshape(ca.x_num, ca.y_num, ca.z_num, 4), 0, 2)
            print(img_vx.shape)
            
            os.makedirs(self.dst, exist_ok=True)
            tifffile.imwrite(self.dst+"/vb_{}_img.tif".format( op), img_vx)
            print("save: ", self.dst+"/vb_{}_img.tif".format( op))
            
            
        else:
            img_vx = tifffile.imread(self.dst+"/vb_{}_img.tif".format( op1))
        
        return img_vx
    
        
    def make_ori_image(self, rID):
        #original mask image
        alpha2=20
        ind = ca.get_vx_ind(rID)

        r_img = np.zeros(len(ca.voxel_ID_order_all), dtype="uint8")

        r_img[ind] = 255

        img_a = []
        for i in r_img:
            if i!=0:
                img_a.append((255,255,255,alpha2))
            else:
        #         if op=="fdr":
        #             alpha2=40
        #             img_a.append((0,0,0,alpha2))
        #         else:
                alpha2=140
                img_a.append((255,255,255,0))

        r_img = np.swapaxes(np.array(img_a).reshape(ca.x_num, ca.y_num, ca.z_num, 4), 0, 2)

        # ach = np.full((ca.z_num, ca.y_num, ca.x_num), 100, dtype=np.uint8)
        # r_img = np.stack([r_img, ach], axis=3)
        print(r_img.shape)
        return r_img
    
    def make_mask_image(self, rID):
        ind_img = np.swapaxes((ca.voxel_ID_order_all).reshape(ca.x_num, ca.y_num, ca.z_num), 0, 2)
        r_img = np.zeros((ca.z_num, ca.y_num, ca.x_num, 4), dtype="uint8")
        
        r_img[np.where(ind_img==rID)] = (255,255,255,0)
        r_img[np.where(ind_img!=rID)] = (0,0,0,255)

      
        return r_img
    
    def make_border_image(self, rID):
        #original mask imageb
        edge_mask = tifffile.imread(self.rdir + "{}um/edge_mask.tif".format(self.vx))
        r_img = np.zeros((ca.z_num, ca.y_num, ca.x_num, 4), dtype="uint8")
        alpha2=100
        r_img[np.where(edge_mask>0)] =  (128, 128,128,alpha2)
        r_img[np.where(edge_mask==0)] =  (255,255,255,0)
        r_img[np.where(np.swapaxes(ca.voxel_ID_order_all.reshape(ca.x_num, ca.y_num, ca.z_num), 0, 2)==rID)] =  (255,255,255,100)
        # ach = np.full((ca.z_num, ca.y_num, ca.x_num), 100, dtype=np.uint8)
        # r_img = np.stack([r_img, ach], axis=3)
        print(r_img.shape)
        return r_img
    
    def make_nuc_image(self, rID):
        

        after_ants_file = self.savedir + "/scn_mask_R50.tif"

        with tifffile.TiffFile(after_ants_file) as tif:
            scn_mask = tif.asarray()

        print(np.sum(scn_mask))
        r_img = np.zeros((ca.z_num, ca.y_num, ca.x_num, 4), dtype="uint8")
     
        alpha2=120
        
        r_img[np.where(scn_mask>0)] =  (255,255,255,alpha2)
        r_img[np.where(scn_mask==0)] =  (255,255,255,0)
        
        return r_img
    
    def make_nuc_atlas_image(self, ants_dir_name):
        
#         after_ants_file = '/home/gpu_data/data1/CUBIC_R_atlas_ver4/1st_CT0_01_SYTOX-G_after_ants.tif'
        
        after_ants_file = savedir + "/mean_nuc_img_{}um.tif".format(self.vx)
        # th_intensity = 1300
        #         xmin_scn, xmax_scn =  160, 182
        #         ymin_scn, ymax_scn = 213, 229
        #         zmin_scn, zmax_scn = 182, 192
        with tifffile.TiffFile(after_ants_file) as tif:
            img = tif.asarray()
#         plt.hist(np.ravel(img))
#         plt.show()

        img_mask= img>200
        mask_ind = np.where(img_mask==1)

        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
                # 8ビット深度に変換
        img = img.astype(np.uint8)
        img[mask_ind] += 50


        
        return img
    
    def make_border_nuc_image(self, rID):
        #original mask imageb
        edge_mask = tifffile.imread(self.rdir + "{}um/edge_mask.tif".format(self.vx))
        r_img = np.zeros((ca.z_num, ca.y_num, ca.x_num, 4), dtype="uint8")
        alpha2=100
        r_img[np.where(edge_mask>0)] =  (128, 128,128,alpha2)
        r_img[np.where(edge_mask==0)] =  (255,255,255,0)
#         r_img[np.where(np.swapaxes(ca.voxel_ID_order_all.reshape(ca.x_num, ca.y_num, ca.z_num), 0, 2)==rID)] =  (255,255,255,100)
 
        after_ants_file = self.savedir + "/scn_mask_R50.tif"
        # th_intensity = 1300
#         xmin_scn, xmax_scn =  160, 182
#         ymin_scn, ymax_scn = 213, 229
#         zmin_scn, zmax_scn = 182, 192
        with tifffile.TiffFile(after_ants_file) as tif:
            scn_mask = tif.asarray()
        # print("nuc_image", image.shape)
        # scn_image = image[self.zmin:self.zmax, self.ymin:self.ymax, self.xmin:self.xmax]
        # scn_in_image = np.zeros_like(image)  #original_size
        # scn_in_image[self.zmin:self.zmax, self.ymin:self.ymax, self.xmin:self.xmax] = scn_image
        # scn_mask = np.where(scn_in_image < th_intensity, 0, 1)
        # print(np.sum(scn_mask))

        
        r_img[np.where(scn_mask>0)] =  (255,255,255,alpha2)
#         r_img[np.where(scn_mask==0)] =  (255,255,255,0)
        
        return r_img
        
        
    def overlay_images(self, atlas_mask , ants_dir_name, size, s, mask=True, op1="None", op2="border", op_alpha = True, n=4, b=0.2):
        
        if op2!="nuc_atlas":
            img_vx = self.make_vb_image(op1, op_alpha, n,b)
        else:
            img_vx = self.make_vb_image2(atlas_mask ,"fdr", op_alpha, n,b)
        
        # r_img = self.make_ori_image(rID)
        if op2=="border":
            r_img = self.make_border_image(rID)
        elif op2 == "nuc":
            r_img = self.make_nuc_image(rID)
        elif op2=="border_nuc":
            r_img = self.make_border_nuc_image(rID)
        elif op2=="nuc_atlas":
            r_img = self.make_nuc_atlas_image(ants_dir_name)
            
        elif op2 == "mask":
            r_img = self.make_mask_image(rID)
        else:
            r_img = self.make_ori_image(rID)
            
        # if "whole" in self.vb_pre_file:
        # os.makedirs(self.dst, exist_ok=True)
        # tifffile.imwrite(self.dst+"/vb_{}_img.tif".format( op1), img_vx)
        # print("save: ", self.dst+"/vb_{}_img.tif".format( op1))
        
        sl_num = size * size + 2
#         mask =True

#         s=0
        if op2!="nuc_atlas":
            fig = plt.figure(figsize=(20,20))
            slice_vs = np.linspace(self.zmin, self.zmax, sl_num)
            for i,sl in enumerate(slice_vs):
                if i==0 or i == len(slice_vs)-1:
                    continue
                ax = fig.add_subplot(size, size, i)
            #     ax.imshow(r_img[int(sl), :,:])
                ax.imshow(img_vx[int(sl), :,:])
                if mask==True:
                    ax.imshow(r_img[int(sl), :,:])

                ax.set_xlim(self.xmin+s, self.xmax-s)
                ax.set_ylim(self.ymax+s, self.ymin-s)
                ax.axis('off')  # 軸を非表示にする


            plt.tight_layout()
            if not os.path.exists(self.dst):
                try:
                    os.makedirs(self.dst)
                except:
                    print("already ex")
            if op_alpha==False:
                fig.savefig(self.dst+"vb_{}_{}_slice_hor.png".format(op1, op2))
            else:
                fig.savefig(self.dst+"vb_{}_{}_alphan{}_b{}_slice_hor.png".format(op1, op2, n, b))
            # plt.show()

            fig = plt.figure(figsize=(20,20))
            slice_vs = np.linspace(self.ymin, self.ymax, sl_num)
            for i,sl in enumerate(slice_vs):
                if i==0 or i == len(slice_vs)-1:
                    continue
                ax = fig.add_subplot(size, size, i)

                ax.imshow(img_vx[:, int(sl),:])
                if mask==True:
                    ax.imshow(r_img[:,int(sl),:])
                ax.axis('off')  # 軸を非表示にする
                ax.set_ylim(self.zmax+s, self.zmin-s)
                ax.set_xlim(self.xmin+s, self.xmax-s)
            plt.tight_layout()


            if op_alpha==False:
                fig.savefig(self.dst+"vb_{}_{}_slice_cor.png".format(op1, op2))
            else:
                fig.savefig(self.dst+"vb_{}_{}_alphan{}_b{}_slice_cor.png".format(op1, op2, n, b))
            # p

            # plt.show()

            fig = plt.figure(figsize=(20,20))
            slice_vs = np.linspace(self.xmin, self.xmax, sl_num)
            for i,sl in enumerate(slice_vs):
                if i==0 or i == len(slice_vs)-1:
                    continue
                ax = fig.add_subplot(size, size, i)
            #     ax.imshow(r_img[:,:,int(sl)])
                ax.imshow(img_vx[:, :,int(sl)])
                if mask==True:
                    ax.imshow(r_img[:,:,int(sl)])
                ax.axis('off')  # 軸を非表示にする
                ax.set_ylim(self.zmax+s, self.zmin-s)
                ax.set_xlim(self.ymin+s, self.ymax-s)
            plt.tight_layout()
            

            if op_alpha==False:
                fig.savefig(self.dst+"vb_{}_{}_slice_sag.png".format(op1, op2))
            else:
                fig.savefig(self.dst+"vb_{}_{}_alphan{}_b{}_slice_sag.png".format(op1, op2, n, b))
            # p

            # plt.show()
        
        else:
            fig = plt.figure(figsize=(20,20))
            slice_vs = np.linspace(self.zmin, self.zmax, sl_num)
            for i,sl in enumerate(slice_vs):
                if i==0 or i == len(slice_vs)-1:
                    continue
                ax = fig.add_subplot(size, size, i)
            #     ax.imshow(r_img[int(sl), :,:])
                ax.imshow(r_img[int(sl), :,:])
                ax.imshow(img_vx[int(sl), :,:])
#                 if mask==True:
                

                ax.set_xlim(self.xmin+s, self.xmax-s)
                ax.set_ylim(self.ymax+s, self.ymin-s)
                ax.axis('off')  # 軸を非表示にする


            plt.tight_layout()
            if not os.path.exists(self.dst):
                try:
                    os.makedirs(self.dst)
                except:
                    print("already ex")

            if op_alpha==False:
                fig.savefig(self.dst+"vb_{}_{}_slice_hor.png".format(op1, op2))
            else:
                fig.savefig(self.dst+"vb_{}_{}_alphan{}_b{}_slice_hor.png".format(op1, op2, n, b))
            # p

            # plt.show()

            fig = plt.figure(figsize=(20,20))
            slice_vs = np.linspace(self.ymin, self.ymax, sl_num)
            for i,sl in enumerate(slice_vs):
                if i==0 or i == len(slice_vs)-1:
                    continue
                ax = fig.add_subplot(size, size, i)

                
#                 if mask==True:
                ax.imshow(r_img[:,int(sl),:])
                ax.imshow(img_vx[:, int(sl),:])
        
                ax.axis('off')  # 軸を非表示にする
                ax.set_ylim(self.zmax+s, self.zmin-s)
                ax.set_xlim(self.xmin+s, self.xmax-s)
            plt.tight_layout()
#             if not os.path.exists(self.savedir+"/region_vb{}_mo{}_r{}/{}/".format(self.vb_r, self.mo, self.r, self.region)):
#                 try:
#                     os.makedirs(self.savedir+"/region_vb{}_mo{}_r{}/{}/".format(self.vb_r, self.mo, self.r, self.region))
#                 except:
#                     print("already ex")

            if op_alpha==False:
                fig.savefig(self.dst+"vb_{}_{}_slice_cor.png".format(op1, op2))
            else:
                fig.savefig(self.dst+"vb_{}_{}_alphan{}_b{}_slice_cor.png".format(op1, op2, n, b))
            # p

            # plt.show()

            fig = plt.figure(figsize=(20,20))
            slice_vs = np.linspace(self.xmin, self.xmax, sl_num)
            for i,sl in enumerate(slice_vs):
                if i==0 or i == len(slice_vs)-1:
                    continue
                ax = fig.add_subplot(size, size, i)
            #     ax.imshow(r_img[:,:,int(sl)])
            
                ax.imshow(r_img[:,:,int(sl)])
                ax.imshow(img_vx[:, :,int(sl)])
#                 if mask==True:
                    
                ax.axis('off')  # 軸を非表示にする
                ax.set_ylim(self.zmax+s, self.zmin-s)
                ax.set_xlim(self.ymin+s, self.ymax-s)
            plt.tight_layout()
#             if not os.path.exists(self.savedir+"/region_vb{}_mo{}_r{}/{}/".format(self.vb_r, self.mo, self.r,self.region)):
#                 try:
#                     os.makedirs(self.savedir+"/region_vb{}_mo{}_r{}/{}/".format(self.vb_r, self.mo,self.r, self.region))
#                 except:
#                     print("already ex")

            if op_alpha==False:
                fig.savefig(self.dst+"vb_{}_{}_slice_sag.png".format(op1, op2))
            else:
                fig.savefig(self.dst+"vb_{}_{}_alphan{}_b{}_slice_sag.png".format(op1, op2, n, b))
            # p

            # plt.show()
    def save_vx_cords_brain(self, atlas_mask):
        if not os.path.exists(self.rdir+"/{}um/voxel_cords_brain.npy".format(self.vx)):
            vx_cords = np.array(np.where(np.swapaxes(atlas_mask, 0, 2)==1)).astype("int32")
            # print(vx_cords)
            # print(type(vx_cords[0][0]))

            # print(vx_cords[0])
            np.save(self.rdir+"/{}um/voxel_cords_brain".format(self.vx), vx_cords.T)



#load atlas data   50um


ca=cfospy.analysis.read_atlas_data(rdir, vx)
print(len(ca.ID_all))
# uni_IDs,rev_IDs = ca.get_uni_rIDs()
# df_sum = ca.get_sum_temp(uni_IDs)
# print(df_sum)

atlas_mask = ca.get_atlas_mask()
print(ca.x_num)
# target_file = "CUBIC-R_Atlas.csv"
# df_atlas_cell = pd.read_csv(rdir + target_file)
# df_iD = pd.read_csv(rdir + sum_file)
# print(df_atlas_cell.iloc[0:2,:])
# atlas_ID_li = df_atlas_cell["atlasID"].tolist()    


if __name__ == '__main__':
    # for rID in rIDs:

        # region = ca.df_allen[ca.df_allen["ID"]==rID]["acronym"].iloc[0]
        region = "whole"

        vb = calc_vb_phase(rdir, savedir, vb_pre_file, atlas_mask,  vx, r, vb_r, mo)
        xmin = vb.xmin
        xmax = vb.xmax
        ymin = vb.ymin
        ymax = vb.ymax
        zmin = vb.zmin
        zmax = vb.zmax
        
        # vb.get_vb_ID()
        # print("x_b_num: " ,vb.x_b_num)
        # print("y_b_num: " ,vb.y_b_num)
        # print("z_b_num: " ,vb.z_b_num)
        # vb.save_vx_cords_brain(atlas_mask)
        vx_cords_f = rdir + "/{}um/voxel_cords_brain.npy".format(vx)
        if not os.path.exists(vb.savedir+ "2nd/"+vb.root_fol_count+ "/CT44_06_vb_CT_{}.bin".format("1st")):
            args = "{} {} {} {} {} {} {} {} {} {}".format(vx, vb_r, mo, vx_cords_f, vb.savedir, vb_pre_file, ants_dir_name, conut_file, ncore, blockdim_x)
            if calc_type=="count" or calc_type=="count_ratio":
                outc = "./vx_c_w " + args
            else:
                outc = "./vx_ci_w " + args
            subprocess.run([outc], shell=True)
      

        vb.make_cos_vb(sample_names)
        # print(cos_v_df)

        vb.overlay_images(atlas_mask, ants_dir_name, size, s, True, "fdr", "nuc_atlas", True, n, b)
        # vb.overlay_images(rID, ants_dir_name, size, s, True, "fdr", "nuc_atlas", False, n, b)
        # vb.overlay_images(rID, ants_dir_name, size, s, True, "fdr", "border", op_alpha, n, b)
        # vb.overlay_images(rID, size, s, mask=True, op1="None", op2="border")
        
        
        # vb.overlay_images(rID, ants_dir_name, size, s, True, "fdr", "mask", op_alpha, n, b)