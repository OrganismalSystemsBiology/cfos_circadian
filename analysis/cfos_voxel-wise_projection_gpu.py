# import ants
from PIL import Image
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
# from joblib import Parallel, delayed
# from skimage import io
import tifffile
from scipy.ndimage import zoom
import traceback
import gzip
import shutil
# from numba import jit
import sys
sys.path.append("/home/gpu_data/data1/kinoshita/")
import cfospy

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from multiprocessing import Pool
plt.gray()

# from costest import costest, batch_costest
import json, os.path, os, re, time, sys
# import joblib
import subprocess
import nibabel as nib
import scipy.spatial
from scipy.signal import convolve
from matplotlib.colors import Normalize, LinearSegmentedColormap, BoundaryNorm, ListedColormap, hsv_to_rgb
import cfospy
from numba import jit

#env CUDA_VISIBLE_DEVICES=0 python3 cfos_voxel-wise_projection_gpu.py
gpu_num = 0

args=sys.argv
alpha = 0.0375  #SEM parameter
r=5
rIDs = [286]#[56, 689, 382, 170, 749, 830, 194, 385] #LGd 170,49664,49668,49672 #HIP382, 423, 726 # DR 872,  VTA 749, TMd 1126,  LHA 194  #  SPZ, DMH, PVN(PVH),PVT   347, 830, 38, 149
l_ID = None #  1097, 549, 1065, 313, 512,   623 698 //315
res = "cos.cell_count_1st2nd_ai_fpr0.5.csv"
# rID = 262
vx  = 20
vb_r=8
angles =["sag"]
size=5   #size*size slice
s=0 #offset image
# op="fdr"  #"None"
ncore = 10
# ants_dir_name = "ANTsR50"
ants_dir_name_point_file = "ANTsR50"
op_alpha = True
n=3
b=0.01

blindness =True
calc_type = "count" 

opi1 = "atlasR{}".format(vx)
opi2 = "RegionPlusBorder200"  #from cropped img
op_pre = "RegionPlusBorder200_lr"


if calc_type == "count":
    calc_fol_l = "region_vb_pro_c_l_blindness2" 
    calc_fol_r = "region_vb_pro_c_r_blindness2" 
elif calc_type == "count_ratio":
    calc_fol_l = "region_vb_pro_cr_l" 
    calc_fol_r = "region_vb_pro_cr_r" 
elif calc_type == "cell_intensity_ratio":
    calc_fol_l = "region_vb_pro_cir_l" 
    calc_fol_r = "region_vb_pro_cir_r"
elif calc_type == "cell_intensity":
    calc_fol_l = "region_vb_pro_ci_l" 
    calc_fol_r = "region_vb_pro_ci_r"

intensity_file= "cell_table_combine_I_ai_fpr0.5.npy"#"cell_intensity_rlratio_img.npy"  # "cell_table_combine_I.npy" #"cell_intensity_norm.npy" #already normed

blockdim_x = blockdim_y = blockdim_z = 8



rdir = "/home/gpu_data/data1/kinoshita/CUBIC_R_atlas_ver5_scn/"
cfos_fol="/home/gpu_data/data1/yamashitaData1/230828circadian_Data1/230828circadian_1st_Reconst/"
exp = "1st"
savedir = "/home/gpu_data/data8/kinoshita_cfos/cfos_app/"

@jit(nopython=True)
def make_edge(rID, mask):

    nonzero_indices = np.nonzero(mask)
    #print(len(nonzero_indices[0]))

    list = []
    for z, y, x in zip(*nonzero_indices):
        if z < 1 or z > mask.shape[0]-2:
            continue
        if y < 1 or y > mask.shape[1]-2:
            continue
        if x < 1 or x > mask.shape[2]-2:
            continue

        is_edge = False
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    if abs(i)+abs(j)+abs(k) == 1:
                        if mask[z+i, y+j, x+k] == 0:
                            is_edge = True
                            break
                if is_edge:
                    break
            if is_edge:
                break
        if is_edge:
            list.append((z, y, x, rID))
    return list

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


# Generate normalized orthogonal basis vectors
def _norm_bases(n_tp, n_tp_in_per):
    """Generate normalized sine and cosine basis vectors for correlation analysis.
    
    Args:
        n_tp (int): Total number of time points in the data
        n_tp_in_per (float): Number of time points in one period
        
    Returns:
        tuple: (normalized_sine_wave, normalized_cosine_wave) - orthogonal basis vectors
    """
    base_s = np.sin(np.arange(n_tp)/n_tp_in_per*2*np.pi)
    base_c = np.cos(np.arange(n_tp)/n_tp_in_per*2*np.pi)
    base_s_m = base_s - np.mean(base_s)
    base_c_m = base_c - np.mean(base_c)
    norm_base_s = (base_s_m)/np.sqrt(np.sum((base_s_m)*(base_s_m)))
    norm_base_c = (base_c_m)/np.sqrt(np.sum((base_c_m)*(base_c_m)))
    return norm_base_s, norm_base_c


# Calculate maximum correlation between input vector and basis vectors
def _calc_max_corr(vec, norm_base_s, norm_base_c):
    """Calculate the maximum correlation between an input vector and sine/cosine basis vectors.
    
    Args:
        vec (array): Input time-series vector
        norm_base_s (array): Normalized sine basis vector
        norm_base_c (array): Normalized cosine basis vector
        
    Returns:
        tuple: (max_correlation_value, phase_of_max_correlation) in radians
    """
    vec_m = vec - np.mean(vec)
    norm_vec = (vec_m)/np.sqrt(np.sum((vec_m)*(vec_m)))
    max_corr_value = np.sqrt(np.power(np.dot(norm_base_c, norm_vec), 2) + np.power(np.dot(norm_base_s, norm_vec), 2))
    # Ensure the maximum correlation does not exceed 1 due to possible numerical subtleties
    max_corr_value = min(1.0, max_corr_value)
    max_corr_phase = np.arctan2(np.dot(norm_base_s, norm_vec), np.dot(norm_base_c, norm_vec))
    return max_corr_value, max_corr_phase


# Calculate p-value for maximum correlation
def _max_corr_pval(num_datapoints, max_corr):
    """Calculate the p-value for the maximum Pearson correlation.
    
    Args:
        num_datapoints (int): Number of data points in the time series
        max_corr (float): Maximum correlation value
        
    Returns:
        float: P-value (probability under null hypothesis)
    """
    n = num_datapoints - 3
    p = np.power((1 - np.power(max_corr, 2)), n / 2)
    return p


# Calculate the SEM adjustment ratio
def _sem_ratio(avg_vector, sem_vector, alpha):
    """Calculate the ratio for adjusting correlation based on standard error of the mean.
    
    This function computes how much the correlation should be adjusted based on the
    confidence interval defined by the SEM values, using a 95% confidence level.
    
    Args:
        avg_vector (array): Average values at each time point
        sem_vector (array): Standard error of the mean values at each time point
        
    Returns:
        float: Adjustment ratio (0.0 to 1.0) for the correlation value
    """
    a_vec = avg_vector - np.mean(avg_vector)
    ratio = 1.0 - alpha / np.sqrt(np.sum(np.power(a_vec / sem_vector, 2))) # 0.67 is the 50% confidence interval
    return max(0, ratio)


# Main function for the analytic cosinor test
def costest(avg_vec, n_tp_in_per, sem_vec=None, alpha=0.67):
    """Perform analytic cosinor test on a time-series vector.
    
    Calculates the maximum Pearson correlation between the input vector and 
    sine/cosine curves with the specified period, along with statistical significance.
    
    Args:
        avg_vec (array): Vector of averaged values at each time point
        n_tp_in_per (float): Number of time points in one period
        sem_vec (array, optional): Vector of standard error of the mean values 
                                  at each time point
    
    Returns:
        tuple: (max_correlation, phase_radians, original_p_value, sem_adjusted_p_value)
            - max_correlation (float): Maximum correlation value
            - phase_radians (float): Phase in radians of the maximum correlation
            - original_p_value (float): Original p-value
            - sem_adjusted_p_value (float): SEM-adjusted p-value
    """
    # Get the number of time points in the data
    n_tp = len(avg_vec)

    # prepare bases
    norm_base_s, norm_base_c = _norm_bases(n_tp, n_tp_in_per)

    # prepare vector to handle nan
    avg_vector = avg_vec.copy()

    # replace nan in AVG with median
    avg_vector[np.isnan(avg_vector)] = np.nanmedian(avg_vector)

    if sem_vec is not None:
        ## prepare vector to handle nan
        sem_vector = sem_vec.copy()

        # each SEM needs to be >0
        sem_vector[~(sem_vector>0)] = np.nanmax(sem_vector)
        # replace nan in SEM with an effectively infinite SEM for missing averages
        sem_vector[np.isnan(avg_vector)] = np.nanmax(sem_vector)*1000000
        # taking the SEM into account
        sem_r = _sem_ratio(avg_vector, sem_vector, alpha)
    else:
        # if no SEM is provided, sem_r is set to 1.0
        sem_r = 1.0

    # tuple of max-correlation value and the phase of max correlation
    mc, mc_ph = _calc_max_corr(avg_vector, norm_base_s, norm_base_c)

    adj_mc = mc * sem_r
    p_org = _max_corr_pval(n_tp, mc)
    p_sem_adj = _max_corr_pval(n_tp, adj_mc)
    
    # max-correlation value, phase of max correlation, original p, SEM adjusted p
    return mc, mc_ph, p_org, p_sem_adj


# Vectorized batch version of the costest function
def batch_costest(avg_vec_matrix, n_tp_in_per, sem_vec_matrix=None, alpha=0.67):
    """Perform analytic cosinor test on multiple time-series vectors.
    
    Calculates the maximum Pearson correlation between each input vector and 
    sine/cosine curves with the specified period, along with statistical significance.
    This vectorized implementation is more efficient for processing large datasets.

    Args:
        avg_vec_matrix (array): 2D array where each row is a vector of averaged 
                               values at each time point
        n_tp_in_per (float): Number of time points in one period
        sem_vec_matrix (array, optional): 2D array where each row is a vector of 
                                         standard error of the mean values at each time point

    Returns:
        ndarray: Array with shape (n_vectors, 4) containing for each input vector:
            - Column 0: Maximum correlation value
            - Column 1: Phase in radians of the maximum correlation
            - Column 2: Original p-value
            - Column 3: SEM-adjusted p-value
    """
    # Convert to numpy array if not already
    avg_vec_matrix = np.asarray(avg_vec_matrix)
    n_vectors, n_tp = avg_vec_matrix.shape

    # prepare bases once
    norm_base_s, norm_base_c = _norm_bases(n_tp, n_tp_in_per)

    # Vectorized NaN handling for avg_vec_matrix
    avg_matrix_clean = avg_vec_matrix.copy()
    nan_mask = np.isnan(avg_matrix_clean)
    # Replace NaN with median for each row
    for i in range(n_vectors):
        if np.any(nan_mask[i]):
            avg_matrix_clean[i, nan_mask[i]] = np.nanmedian(avg_matrix_clean[i])

    # Vectorized correlation calculation
    # Center the data
    avg_means = np.mean(avg_matrix_clean, axis=1, keepdims=True)
    avg_centered = avg_matrix_clean - avg_means
    
    # Normalize the vectors
    avg_norms = np.sqrt(np.sum(avg_centered**2, axis=1, keepdims=True))
    norm_avg = avg_centered / avg_norms
    
    # Calculate correlations with sine and cosine bases
    corr_s = np.dot(norm_avg, norm_base_s)
    corr_c = np.dot(norm_avg, norm_base_c)
    
    # Calculate max correlation and phase
    mc_array = np.sqrt(corr_s**2 + corr_c**2)
    mc_array = np.minimum(mc_array, 1.0)  # Ensure max correlation doesn't exceed 1
    mc_ph_array = np.arctan2(corr_s, corr_c)
    
    # Calculate original p-values vectorized
    p_org_array = np.power((1 - mc_array**2), (n_tp - 3) / 2)
    
    # Handle SEM adjustments vectorized
    sem_r_array = np.ones(n_vectors)
    if sem_vec_matrix is not None:
        sem_vec_matrix = np.asarray(sem_vec_matrix)
        sem_matrix_clean = sem_vec_matrix.copy()
        
        # Vectorized SEM handling
        sem_nonpositive = ~(sem_matrix_clean > 0)
        sem_nan = np.isnan(sem_matrix_clean)
        
        # Replace non-positive and NaN values
        for i in range(n_vectors):
            sem_max = np.nanmax(sem_matrix_clean[i])
            sem_matrix_clean[i, sem_nonpositive[i]] = sem_max
            sem_matrix_clean[i, sem_nan[i]] = sem_max * 1000000
            
            # Calculate SEM ratio
            ratio_vec = avg_centered[i] / sem_matrix_clean[i]
            sem_r_array[i] = max(0, 1.0 - alpha / np.sqrt(np.sum(ratio_vec**2)))
    
    # Calculate SEM-adjusted correlations and p-values
    adj_mc_array = mc_array * sem_r_array
    p_sem_adj_array = np.power((1 - adj_mc_array**2), (n_tp - 3) / 2)

    return np.column_stack([mc_array, mc_ph_array, p_org_array, p_sem_adj_array])

    
import math
def rad2ph(rad):
#     return (int((2*np.pi+rad)*180/np.pi*24/360),  int((rad)*180/np.pi*24/360))[rad>=0]
    if math.isnan(rad):
        return np.nan
    else:
        return (round((2*np.pi+rad)*180/np.pi*24/360, 1),  round((rad)*180/np.pi*24/360, 1))[rad>=0]
    
# cfos_fol="/home/data1/yamashitaData1/231012_circadian_2nd_Data1/231012_circadian_2nd_Reconst/"
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

#load atlas data   50um




class calc_vb_phase:
    def __init__(self, rdir, savedir,calc_fol, rID, region, vx, r, vb_r, angle, xmin, xmax, ymin, ymax, zmin, zmax):
        self.region = region
        self.vx = vx
        self.r = r
        self.savedir = savedir
        self.vb_r = vb_r
     
        self.rdir = rdir
        self.angle = angle
    #     if region != "MED":  # "SCH":
        #     if region != "MED":  # "SCH":
        #         continue
        print(self.region)
        print(rID)


        if angle =="hor":
            self.mox = 1
            self.moy = 1
            self.moz = int(zmax - zmin)
        elif angle =="cor":
            self.mox = 1
            self.moy = int(ymax - ymin)
            self.moz = 1
        elif angle =="sag":
            self.mox = int(xmax-xmin)
            self.moy = 1
            self.moz = 1
        
        
        
        vol = (xmax-xmin)*(ymax-ymin)*(zmax-zmin)
        print("voxel volume", vol)
        
       
        
      
#         return xmin, xmax, ymin, ymax, zmin, zmax
#         if calc_type =="count_ratio":
#             calc_fol = "region_vb"#cell count
#         elif calc_type =="count":

        #
       
        self.calc_fol = calc_fol
        self.fol = "/"+calc_fol+"/{}um/{}/vb{}_mo{}_{}_{}_r{}/".format(self.vx, self.region, self.vb_r, self.mox, self.moy, self.moz, self.r)
        
        self.root_fol_count="/"+ calc_fol + "/{}um/{}/vb{}_mo{}_{}_{}_r{}/".format(self.vx, self.region, self.vb_r, self.mox, self.moy, self.moz,  self.r)

        
        self.dst = self.savedir + self.fol 
        os.makedirs(self.savedir+self.fol, exist_ok=True)
        print("make", self.savedir+self.fol)
#         self.mo = int(round((math.pow(vol/190000, 1/3)), 0) + 1)
#         print("moving", self.mo)

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax

#         return xmin, xmax, ymin, ymax, zmin, zmax

    
    def get_vb_ID(self):
        img_vx_ID = np.arange(0, len(ca.voxel_ID_order_all)).reshape(ca.x_num, ca.y_num, ca.z_num)
        # print("img_vx_ID", img_vx_ID.shape)
        # print(ca.x_num)
        # print(ca.y_num)
        # print(ca.z_num)
        vb_ID = {}
        c = 0
        
        if self.angle=="hor":
            self.x_b_num = int((self.xmax-self.xmin)/self.mox+1)
            self.y_b_num = int((self.ymax-self.ymin)/self.moy+1)
            self.z_b_num = int((self.zmax-self.zmin)/self.moz)
        elif self.angle=="cor":
            self.x_b_num = int((self.xmax-self.xmin)/self.mox+1)
            self.y_b_num = int((self.ymax-self.ymin)/self.moy)
            self.z_b_num = int((self.zmax-self.zmin)/self.moz+1)
        elif self.angle=="sag":
            self.x_b_num = int((self.xmax-self.xmin)/self.mox)
            self.y_b_num = int((self.ymax-self.ymin)/self.moy+1)
            self.z_b_num = int((self.zmax-self.zmin)/self.moz+1)
        self.total_b_num = self.x_b_num*self.y_b_num*self.z_b_num
        
        if not os.path.exists(self.savedir+self.fol+"vb_ID.pkl"):
            for x in range(self.x_b_num):
                for y in range(self.y_b_num):
                    for z in range(self.z_b_num):
    #                     if self.xmin+x*self.mo-int(self.vb_r/2) <0:
    #                         xmin2 = 0
    #                     else:
    #                         xmin2 =  self.xmin+x*self.mo-int(self.vb_r/2)

    #                     if self.xmin+x*self.mo+int(self.vb_r/2) > ca.x_num:
    #                         xmax2 = ca.x_num
    #                     else:
    #                         xmax2 = self.xmin+x*self.mo+int(self.vb_r/2)

    #                     if self.ymin+y*self.mo-int(self.vb_r/2) <  0:
    #                         ymin2=0
    #                     else:
    #                         ymin2 = self.ymin+y*self.mo-int(self.vb_r/2)

    #                     if self.ymin+y*self.mo+int(self.vb_r/2) >ca.y_num:
    #                         ymax2=ca.y_num
    #                     else:
    #                         ymax2 = self.ymin+y*self.mo+int(self.vb_r/2)

    #                     if self.zmin+z*self.mo-int(self.vb_r/2) < 0:
    #                         zmin2 = 0
    #                     else:
    #                         zmin2 = self.zmin+z*self.mo-int(self.vb_r/2)

    #                     if self.zmin+z*self.mo+int(self.vb_r/2) > ca.z_num:
    #                         zmax2 = ca.z_num
    #                     else:
    #                         zmax2 = self.zmin+z*self.mo+int(self.vb_r/2)

    #                         print(xmin2)

                        if self.angle=="hor":
                            if self.xmin+x*self.mox >= img_vx_ID.shape[0]:
                                xv = img_vx_ID.shape[0]-1
                            else:
                                xv = self.xmin+x*self.mox

                            if self.ymin+y*self.moy >= img_vx_ID.shape[1]:
                                yv = img_vx_ID.shape[1]-1
                            else:
                                yv = self.ymin+y*self.moy

                            if self.zmax >= img_vx_ID.shape[2]:
                                zv = img_vx_ID.shape[2]
                            else:
                                zv = self.zmax
                            
                            vb_ID[c] = img_vx_ID[xv, yv, self.zmin:zv]
                        elif self.angle=="cor":
                            if self.xmin+x*self.mox >= img_vx_ID.shape[0]:
                                xv = img_vx_ID.shape[0]-1
                            else:
                                xv = self.xmin+x*self.mox

                            if self.zmin+z*self.moz >= img_vx_ID.shape[2]:
                                zv = img_vx_ID.shape[2]-1
                            else:
                                zv = self.zmin+z*self.moz
                            if self.ymax >= img_vx_ID.shape[1]:
                                yv = img_vx_ID.shape[1]
                            else:
                                yv = self.ymax
                                
                            # print("xv", xv)
                            # print("yv", yv)
                            # print("zv", zv)

                            
                            vb_ID[c] = img_vx_ID[xv, self.ymin:yv, zv]


                        elif self.angle=="sag":
                            if self.ymin+y*self.moy >= img_vx_ID.shape[1]:
                                yv = img_vx_ID.shape[1]-1
                            else:
                                yv = self.ymin+y*self.moy

                            if self.zmin+z*self.moz >= img_vx_ID.shape[2]:
                                zv = img_vx_ID.shape[2]-1
                            else:
                                zv = self.zmin+z*self.moz

                            if self.xmax > img_vx_ID.shape[0]:
                                xv = img_vx_ID.shape[0]
                            else:
                                xv = self.xmax
                            
                            # print("xv", xv)
                            # print("yv", yv)
                            # print("zv", zv)
                            
                            vb_ID[c] = img_vx_ID[self.xmin:xv, yv, zv]
                        c+=1





                with open(self.savedir+self.fol+"vb_ID.pkl", "wb") as tf:
                    pickle.dump(vb_ID, tf)
#         return vb_ID

        
    def make_cos_vb(self, sample_names):
        
        # if not os.path.exists(self.dst + "/cos_1st2nd.csv"):
            CT_li = np.arange(0, 48, 4)
            CT_li2 = np.arange(0, 96, 4)
            sample_ids = np.arange(1, 7, 1)
            # for l, exp in enumerate(exps):
            
            
            vb_exp = np.zeros((self.total_b_num, len(CT_li)*len(sample_ids)), dtype = "float32")
            exp = "1st"
            os.makedirs(self.savedir+exp+ "/"+self.fol, exist_ok=True)
            
            if not os.path.exists(self.savedir+exp+"/"+self.fol + "vb_CT_{}".format(exp)):
                for m, CT in enumerate(CT_li):
                    for n,sample_id in enumerate(sample_ids):
                        sample = "CT"+str(CT)+"_"+str(sample_id).zfill(2)

                        cf =self.savedir+exp+ "/"+self.root_fol_count+ "/"+sample +"_vb_CT_{}.bin".format(exp)
                        rectype = np.dtype(np.int32)
                        vb_bin = np.fromfile(cf, dtype=rectype)#.astype("float32")
                        vb_exp[:,m*len(sample_ids)+n] = vb_bin
                if calc_type == "cell_intensity_ratio":
                    T_cells = np.load(self.savedir+exp+"/total_cell_intenses.npy")
                    vb_exp = vb_exp/T_cells.reshape(-1, len(CT_li)*len(sample_ids))  #ratio
                elif calc_type == "count_ratio":
                    T_cells = np.load(self.savedir+exp+"/total_cell_nums.npy")
                    vb_exp = vb_exp/T_cells.reshape(-1, len(CT_li)*len(sample_ids)) 
                else:
                    vb_exp = vb_exp
                
                np.save(self.savedir+exp+"/"+self.fol + "vb_CT_{}".format(exp), vb_exp)
            
            
            
            exp = "2nd"
            os.makedirs(self.savedir+exp+ "/"+self.fol, exist_ok=True)
            if not os.path.exists(self.savedir+exp+"/"+self.fol + "vb_CT_{}".format(exp)):
                for m, CT in enumerate(CT_li):
                    for n,sample_id in enumerate(sample_ids):
                        sample = "CT"+str(CT)+"_"+str(sample_id).zfill(2)

                        cf =self.savedir+exp+ "/"+self.root_fol_count + "/"+sample +"_vb_CT_{}.bin".format(exp)
                        rectype = np.dtype(np.int32)
                        vb_bin = np.fromfile(cf, dtype=rectype)#.astype("float32")
                        vb_exp[:,m*len(sample_ids)+n] = vb_bin
                        
                if calc_type == "cell_intensity_ratio":
                    T_cells = np.load(self.savedir+exp+"/total_cell_intenses.npy")
                    vb_exp = vb_exp/T_cells.reshape(-1, len(CT_li)*len(sample_ids))  #ratio
                elif calc_type == "count_ratio":
                    T_cells = np.load(self.savedir+exp+"/total_cell_nums.npy")
                    vb_exp = vb_exp/T_cells.reshape(-1, len(CT_li)*len(sample_ids)) 
                else:
                    vb_exp = vb_exp
                
                np.save(self.savedir+exp+"/"+self.fol + "vb_CT_{}".format(exp), vb_exp)
            
            

            df_all=[]
            df_all_r=[]
            cols =[]
            
            for m, CT in enumerate(CT_li2):
                for n,sample_id in enumerate(sample_ids):
                    sample = "CT"+str(CT)+"_"+str(sample_id).zfill(2)
                    cols.append(sample)
            
            path1 = self.savedir+"1st/"+self.fol+"/" + "vb_CT_{}.npy".format("1st")
            CT_np1 = np.load(path1)
                
                
            path2 = self.savedir+"2nd/"+ self.fol+"/"+ "vb_CT_{}.npy".format("2nd")
            CT_np2 = np.load(path2)
            

            df = np.hstack([CT_np1, CT_np2])
            del CT_np1
            del CT_np2
            gc.collect()

            df = pd.DataFrame(df)
            df.columns=cols
            df.insert(0, "id", np.arange(0, self.total_b_num))
            print(df)


            avg_list = []
            se_list = []
            for i in range(df.shape[0]):
                vals = df.iloc[i, 1:(1+12*2*6)].to_numpy(dtype=int)
                tbl = vals.reshape(24, 6).astype("uint32")
                avg = tbl.mean(axis=1).astype("float32")
                se = (tbl.std(axis=1,ddof=1) / np.sqrt(tbl.shape[1])).astype("float32")
                avg_list.append(avg)
                se_list.append(se)
            
            del vals
            del tbl
            del avg
            del se
            gc.collect()
            # Batch analysis for multiple time-series
            data_matrix = np.array(avg_list).astype("float32")  # Multiple time-series
            se_matrix = np.array(se_list).astype("float32")  # Corresponding SEMs
            del avg_list
            del se_list
            gc.collect()

            results = (batch_costest(data_matrix[:,:], 6, se_matrix, alpha)).astype("float32")

            del data_matrix
            del se_matrix
            gc.collect()

            id_list = df["id"].tolist()
            
            del df
            gc.collect()



            mc, mc_ph, p_org, p_sem_adj =results[:,0],results[:,1],results[:,2],results[:,3]
            nan_id = np.isnan(p_sem_adj).astype("uint32")
            nonan_id = (np.where(nan_id==False)[0]).astype("uint32")
            del nan_id
            del p_org
            
            gc.collect()
            
            phase_li = list(map(rad2ph, mc_ph))
           # acronyms= [ca.df_allen[ca.df_allen["ID"]==rID]["acronym"].iloc[0] for rID in df["id"]]
#node_names= [ca.df_allen[ca.df_allen["ID"]==rID]["node_name"].iloc[0] for rID in df["id"]]
            per_li = np.ones(len(phase_li), dtype="uint8")*24


           
            # cos_v_df = pd.DataFrame({"id":df["id"].tolist(),  "ADJ.P":p_sem_adj,  "PER":per_li, "Ph":mc_ph, "LAG":phase_li, "max_corr":mc,})
            
            cos_v_df = pd.DataFrame({"id":id_list,  "ADJ.P":p_sem_adj,  "PER":per_li, "Ph":mc_ph, "LAG":phase_li, "max_corr":mc,})
            

            del per_li 
        
            del phase_li
            del mc
            del mc_ph
            gc.collect()

            p_bh_np = np.ones(len(p_sem_adj), dtype="float32")
            bool_, p_bh_li, _, _ =  multipletests(p_sem_adj[nonan_id],  method="fdr_bh")
            # bool_, p_bh_li, _, _ =  multipletests(p_sem_adj[~np.isnan(p_sem_adj)],  method="fdr_bh")
            del bool_
            del p_sem_adj
            gc.collect()

            p_bh_np[nonan_id] = p_bh_li

            del p_bh_li
            del nonan_id
        
            gc.collect()
            
            cos_v_df.insert(3, "BH.Q", p_bh_np)

            del p_bh_np
            gc.collect()
            

            # cos_v_df = cos_v_df.sort_values("BH.Q")

            cos_v_df.to_csv(self.dst+"cos_1st2nd.csv")
#         else:
#             cos_v_df = pd.read_csv(self.savedir+self.fol+"cos_1st2nd.csv", index_col=0)
        
#         return cos_v_df
    
        
    def make_vb_image(self, op="None", op_alpha = True, n=4, b=0.2):
#         op ="fdr"
        # op="None"

        

        
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
    
    def make_vb_image2(self, op="None", op_alpha = True, n=4, b=0.1):
#         op ="fdr"
        # op="None"
#         n = 6
        a = (1-b)/(-1)**n
# y=b*(math.e)**(np.log(1/b)*a*x)

        # if not os.path.exists(self.savedir +self.fol +  "{}_img_{}.tif".format(op, self.angle)):
        

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
#                 alpha_li = np.clip(alpha_li, a_lim, 1)*255

        if blindness == True:
            ph_li = (cos_v_df["LAG"]/24)
            # ph_li = 1-ph_li
            # ph_li = [(hue + 1/3 - 1) if (hue + 1/3) > 1 else (hue + 1/3) for hue in ph_li]

            

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
                        hsv_li.append(np.append(np.array(cmap_icefire(i))[0:3]*255, 255))
                    elif math.isnan(i) or fdr_li_all[n]!=1:
                        hsv_li.append((0,0,0, 0))  #black  #解析したけど振動なし、phaseなし
                    else:
                        hsv_li.append((0,0,0, 0)) #black
                elif op=="fdr" and op_alpha ==True:
                    if not math.isnan(i) and not i==-1 and fdr_li_all[n]==1:
                        hsv_li.append(np.append(np.array(cmap_icefire(i))[0:3]*255, alpha_li_all[n]))
                #         print(hsv_to_rgb([i, 1, 1]))
                    elif math.isnan(i) or fdr_li_all[n]!=1:
            #             hsv_li.append((255,255,255, alpha))  #white  #解析したけど振動なし、phaseなし
                        hsv_li.append((0,0,0, 0))  #black  #解析したけど振動なし、phaseなし
                    else:
                        hsv_li.append((0,0,0, 0)) #black

                else:
                    if not math.isnan(i) and not i==-1:
                        hsv_li.append(np.append(np.array(cmap_icefire(i))[0:3]*255, alpha))
                #         print(hsv_to_rgb([i, 1, 1]))
                    elif math.isnan(i):
            #             hsv_li.append((255,255,255, alpha))  #white  #解析したけど振動なし、phaseなし
                        hsv_li.append((0,0,0, alpha))  #black  #解析したけど振動なし、phaseなし
                    else:
                        hsv_li.append((0,0,0, alpha)) #black
        else:

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
                    elif math.isnan(i) or fdr_li_all[n]!=1:
                        hsv_li.append((0,0,0, 0))  #black  #解析したけど振動なし、phaseなし
                    else:
                        hsv_li.append((0,0,0, 0)) #black
                elif op=="fdr" and op_alpha ==True:
                    if not math.isnan(i) and not i==-1 and fdr_li_all[n]==1:
                        hsv_li.append(np.append(hsv_to_rgb([i, 1, 1])*255, alpha_li_all[n]))
                #         print(hsv_to_rgb([i, 1, 1]))
                    elif math.isnan(i) or fdr_li_all[n]!=1:
            #             hsv_li.append((255,255,255, alpha))  #white  #解析したけど振動なし、phaseなし
                        hsv_li.append((0,0,0, 0))  #black  #解析したけど振動なし、phaseなし
                    else:
                        hsv_li.append((0,0,0, 0)) #black

                else:
                    if not math.isnan(i) and not i==-1:
                        hsv_li.append(np.append(hsv_to_rgb([i, 1, 1])*255, alpha))
                #         print(hsv_to_rgb([i, 1, 1]))
                    elif math.isnan(i):
            #             hsv_li.append((255,255,255, alpha))  #white  #解析したけど振動なし、phaseなし
                        hsv_li.append((0,0,0, alpha))  #black  #解析したけど振動なし、phaseなし
                    else:
                        hsv_li.append((0,0,0, alpha)) #black

        print(np.array(hsv_li).shape)

        hsv_li = np.array(hsv_li).astype("uint8")

        img_vx = np.swapaxes(hsv_li.reshape(ca.x_num, ca.y_num, ca.z_num, 4), 0, 2)
        print(img_vx.shape)
        
        tifffile.imwrite(self.savedir +self.fol +  "{}_img_{}_new.tif".format(op, self.angle), img_vx)
        
        # else:
        #     img_vx = tifffile.imread(self.savedir +self.fol +  "{}_img_{}.tif".format(op, self.angle))
            
        return img_vx
    
    
    def make_vb_image_edge(self, rID, img_vx, lr, op1, op2):
        # opi1 = "atlasR{}".format(self.vx)
        # opi2 = "RegionPlusBorder0.5"
        
        # if not os.path.exists(self.savedir+self.fol+"vb_{}_{}_boder_slice_{}.png".format(op1, op2,  self.angle)):
        
            # if not os.path.exists(self.savedir+self.fol+"vb_{}_boder_slice_{}.tif".format(op1,  self.angle)):
                # if not os.path.exists(savedir + "region_edge/"+ "edge_{}_{}um_{}.tif".format( self.region, vx, self.r)):
                r_edge = np.zeros((ca.z_num, ca.y_num, ca.x_num, 4), dtype="uint8")
                edge_mask = tifffile.imread(rdir +"{}um/edge_mask.tif".format(vx))
                edge_vx_ind = np.where(edge_mask==1)
                r_edge[edge_vx_ind] = (255,255,255, 100) #other regions border
                edge_list = make_edge(rID, mask0)
                for z, y, x, _ in edge_list:
                    # r_edge[z, y, x] = (139, 69, 19, 255)  #border of region of interest
                    r_edge[z, y, x] = (255, 255,255, 255) 
                r_edge[edge_mask==0] ==  (0,0,0, 0) #non edge voxel
                #     os.makedirs(savedir + "region_edge/", exist_ok=True)
                #     tifffile.imwrite(savedir + "region_edge/"+ "edge_{}_{}um_{}.tif".format(self.region, vx, self.r), r_edge)
                # else:
                #     r_edge = tifffile.imread(savedir + "region_edge/"+ "edge_{}_{}um_{}.tif".format( self.region, vx, self.r))
                
                center_l, center_r = ca.get_lr_center(rID)
                
                if lr == "left":
                    center = center_l
                    r_img = tifffile.imread(self.savedir + "1st" + "/region_crop_atlasR{}_{}_{}_maxp_mean_".format(self.vx, op_pre.replace("_lr",""), "left")+self.angle+"/" + self.region + "/"+"CT0"+".tif")
                
                else:
                    center = center_r
                    r_img = tifffile.imread(self.savedir + "1st" + "/region_crop_atlasR{}_{}_{}_maxp_mean_".format(self.vx, op_pre.replace("_lr",""), "right")+self.angle+"/" + self.region + "/"+"CT0"+".tif")
                
                
                
                if self.angle == "hor":
                    img_vx_c = img_vx[self.zmin, int(self.ymin):int(self.ymax), int(self.xmin):int(self.xmax),:]
                    r_edge = r_edge[int(center[0]), int(self.ymin):int(self.ymax), int(self.xmin):int(self.xmax),:]
                    # width =  img_vx_c.shape[1]
                    # height =  img_vx_c.shape[0]
                    # base = 1/ca.x_num*width
                elif self.angle == "cor":
                    img_vx_c = img_vx[int(self.zmin):int(self.zmax), self.ymin ,int(self.xmin):int(self.xmax),:]
                    r_edge = r_edge[int(self.zmin):int(self.zmax), int(center[1]), int(self.xmin):int(self.xmax),:]
                    # width =  img_vx_c.shape[1]
                    # height =  img_vx_c.shape[0]
                    # base = 1/ca.x_num*width
                elif self.angle == "sag":
                    img_vx_c = img_vx[int(self.zmin):int(self.zmax), int(self.ymin):int(self.ymax), self.xmin,:]
                    r_edge = r_edge[int(self.zmin):int(self.zmax), int(self.ymin):int(self.ymax), int(center[2]),:]
                    # width =  img_vx_c.shape[1]
                    # height =  img_vx_c.shape[0]
                    # base = 1/ca.y_num*width
                # img_vx_c[np.where(r_edge !=(0,0,0,0))[0:2]]=r_edge[np.where(r_edge !=(0,0,0,0))[0:2]]
                
                # del r_edge
                # gc.collect()
                
            #     tifffile.imwrite(self.savedir+self.fol+"vb_{}_slice_{}.tif".format(op1,  self.angle), img_vx_c)
            # else:
            #     img_vx_c = tifffile.imread(self.savedir+self.fol+"vb_{}_slice_{}.tif".format(op1,  self.angle))
            
            # r_img = self.make_nuc_atlas_image()
            # if self.angle == "hor":
            #     r_img = np.max(r_img, axis = 0)[int(self.ymin):int(self.ymax), int(self.xmin):int(self.xmax)]
            # elif self.angle == "cor":
            #     r_img = np.max(r_img, axis = 1)[int(self.zmin):int(self.zmax), int(self.xmin):int(self.xmax)]
            # elif self.angle == "sag":
            #     r_img = np.max(r_img, axis = 2)[int(self.zmin):int(self.zmax), int(self.ymin):int(self.ymax)]
            # r_img = ((r_img - np.min(r_img)) / (np.max(r_img) - np.min(r_img)) * 255).astype(np.uint8)
            # rgb_img = np.zeros((r_img.shape[0],r_img.shape[1], 4), dtype="uint8")
            # rgb_img[:,:,0] = r_img
            # rgb_img[:,:,1] = r_img
            # rgb_img[:,:,2] = r_img
            # rgb_img[:,:,3] = 255
            # del r_img
            # gc.collect()
            
            
                # if os.path.exists(self.savedir + "1st" + "/region_crop_{}_{}_{}_maxp_mean_".format(opi1,opi2,lr)+self.angle+"/" + self.region + "/"+"CT"+str(0) +".tif"):
                    # rgb_img =  tifffile.imread(   self.savedir + "1st" + "/region_crop_{}_{}_{}_maxp_mean_".format(opi1,opi2,lr)+self.angle+"/" + self.region + "/"+"CT"+str(0) +".tif") 
                # r_img = tifffile.imread(self.savedir+exp+"/"+"CT0_01"+"/cfos/"+"ANTsR{}".format(self.vx) + "/after_ants.tif")
                # r_img = tifffile.imread(self.savedir+"1st"+"/"+"CT0_01"+"/cfos/"+"ANTsR{}".format(self.vx) + "/after_ants.tif")
                # r_img = tifffile.imread(self.savedir+"/mean_cfos_img_{}um.tif".format(vx))
                
                # if self.angle == "hor":
                #     r_img = np.max(r_img, axis=0)[int(self.ymin):int(self.ymax), int(self.xmin):int(self.xmax)]
                #     width =  r_img.shape[1]
                #     height =  r_img.shape[0]
                #     base = 1/ca.x_num*width
                # elif self.angle == "cor":
                #     r_img = np.max(r_img, axis=1)[int(self.zmin):int(self.zmax) ,int(self.xmin):int(self.xmax)]
                #     width =  r_img.shape[1]
                #     height =  r_img.shape[0]
                #     base = 1/ca.x_num*width
                # elif self.angle == "sag":
                #     r_img = np.max(r_img, axis=2)[int(self.zmin):int(self.zmax), int(self.ymin):int(self.ymax)]
                #     width =  r_img.shape[1]
                #     height =  r_img.shape[0]
                #     base = 1/ca.y_num*width
                
             
                print("r_img", r_img.shape)
                print("r_edge", r_edge.shape)
                print("img_vx_c", img_vx_c.shape)

                # if lr == "right":
                # r_img = np.flip(r_img, axis=1).copy()


                # if base < 0.05:
                #     var = 0.05
                # elif 0.05<= base < 0.1:
                #     var = 0.1
                # elif 0.1<=base and base < 0.2:
                #     var = 0.2
                # elif base >= 0.2 and base <0.5:
                #     var = 0.5
                # elif base >= 0.5 and base <1.0:
                #     var = 1.0
                # else:
                #     var=1.0

               
                plt.figure(figsize = (5,5))
                plt.imshow(r_img)
                plt.imshow(img_vx_c)
                plt.imshow(r_edge)
                plt.axis("off")
                
                # scalebar_length =  var*1000/vx #mm   #x_num * 0.1  # distance in img
                # scalebar_position = (width - scalebar_length - int(15/147*width) , height-int(2/37*height))
                # # スケールバーを追加します。
                # scalebar = patches.Rectangle(scalebar_position, scalebar_length, 0.8/93*height, color='brown')
                # plt.gca().add_patch(scalebar)
                # # スケールバーの近くにテキストを追加します。
                # text_position = (scalebar_position[0]  + scalebar_length / 2 , scalebar_position[1]-int(2/45*height) )
                # plt.text(*text_position, '{} mm'.format(var), color='brown', ha='center')
                
                if blindness ==False:
                    plt.savefig(self.savedir+self.fol+"vb_{}_{}_boder_slice_{}2_new.png".format(op1, op2,  self.angle))
                    plt.savefig(self.savedir+self.fol+"vb_{}_{}_boder_slice_{}2_new.SVG".format(op1, op2,  self.angle))
                                

                    # tifffile.imwrite(self.savedir+self.fol+"vb_{}_{}_boder_slice_{}.tif".format(op1, op2,  self.angle), rgb_img)
                    print("save", self.savedir+self.fol+"vb_{}_{}_boder_slice_{}_new.png".format(op1, op2,  self.angle))
                else:
                    plt.savefig(self.savedir+self.fol+"vb_{}_{}_boder_slice_{}2_new_bc.png".format(op1, op2,  self.angle))
                    plt.savefig(self.savedir+self.fol+"vb_{}_{}_boder_slice_{}2_new_bc.SVG".format(op1, op2,  self.angle))
                                

                    # tifffile.imwrite(self.savedir+self.fol+"vb_{}_{}_boder_slice_{}.tif".format(op1, op2,  self.angle), rgb_img)
                    print("save", self.savedir+self.fol+"vb_{}_{}_boder_slice_{}_new_bc.png".format(op1, op2,  self.angle))
                
                #     print("cfos_crop_region is not found")
        
        # else:
        #     print("{} is already exist ".format(self.savedir+self.fol+"vb_{}_{}_boder_slice_{}.png".format(op1, op2,  self.angle)))
    
    
        
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
    
    def make_nuc_atlas_image(self):
        
        after_ants_file = self.savedir + "1st/CT0_01/cfos/ANTsR20/after_ants.tif"
        
        # after_ants_file = savedir + "/mean_nuc_img_{}um.tif".format(self.vx)
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
        
        
    def overlay_images(self, rID, size, s, mask=True, op1="None", op2="border", op_alpha = True, n=4, b=0.2):
        
        if op2!="nuc_atlas":
            img_vx = self.make_vb_image(op1, op_alpha, n,b)
        else:
            img_vx = self.make_vb_image2("fdr", op_alpha, n,b)
        
#         r_img = self.make_ori_image(rID)
        if op2=="border":
            r_img = self.make_border_image(rID)
        elif op2 == "nuc":
            r_img = self.make_nuc_image(rID)
        elif op2=="border_nuc":
            r_img = self.make_border_nuc_image(rID)
        elif op2=="nuc_atlas":
            r_img = self.make_nuc_atlas_image()
        elif op2 == "mask":
            r_img = self.make_mask_image(rID)
        else:
            r_img = self.make_ori_image(rID)
        
        
        sl_num = size * size + 2
#         mask =True

#         s=0
        if op2!="nuc_atlas":
            fig = plt.figure(figsize=(20,20))
            ax = fig.add_subplot(1,1,1)

            if self.angle=="hor":
#                 ax.imshow(np.max(r_img[self.zmin:self.zmax, :,:], axis=0))
                ax.imshow(img_vx[self.zmin + 5, :,:])
                ax.imshow(r_img[self.zmin + 5, :,:])
        
                ax.set_xlim(self.xmin+s, self.xmax-s)
                ax.set_ylim(self.ymax-s, self.ymin+s)
            elif self.angle=="cor":
#                 ax.imshow(np.max(r_img[:, self.ymin:self.ymax, :], axis=0))
                ax.imshow(img_vx[:, self.ymin+5, :])
                ax.imshow(r_img[:, self.ymin+5, :])
                ax.set_ylim(self.zmax-s, self.zmin+s)
                ax.set_xlim(self.xmin+s, self.xmax-s)
            elif self.angle=="sag":
#                 ax.imshow(np.max(r_img[:, :, self.xmin:self.xmax], axis=0))
                ax.imshow(img_vx[:, :, self.xmin+5])
                ax.imshow(r_img[:, :, self.xmin+5])
                ax.set_ylim(self.zmax-s, self.zmin+s)
                ax.set_xlim(self.ymin+s, self.ymax-s)

            ax.axis('off')  # 軸を非表示にする
            plt.tight_layout()
            
            if op_alpha==False:
                fig.savefig(self.savedir+self.fol+"vb_{}_{}_slice_{}.png".format(op1, op2, self.angle))
            else:
                fig.savefig(self.savedir+self.fol+"vb_{}_{}_alphan{}_b{}_slice_{}_new.png".format(op1, op2, n,b, self.angle))

            plt.show()

        
        else:
        
            fig = plt.figure(figsize=(20,20))
            ax = fig.add_subplot(1,1,1)

            if self.angle=="hor":
                ax.imshow(np.max(r_img[self.zmin:self.zmax, :,:], axis=0))
                ax.imshow(img_vx[self.zmin + 5, :,:])
                ax.set_xlim(self.xmin+s, self.xmax-s)
                ax.set_ylim(self.ymax-s, self.ymin+s)
            elif self.angle=="cor":
                ax.imshow(np.max(r_img[:, self.ymin:self.ymax, :], axis=1))
                ax.imshow(img_vx[:, self.ymin+5, :])
                ax.set_ylim(self.zmax-s, self.zmin+s)
                ax.set_xlim(self.xmin+s, self.xmax-s)
            elif self.angle=="sag":
                ax.imshow(np.max(r_img[:, :, self.xmin:self.xmax], axis=2))
                ax.imshow(img_vx[:, :, self.xmin+5])
                ax.set_ylim(self.zmax-s, self.zmin+s)
                ax.set_xlim(self.ymin+s, self.ymax-s)

            ax.axis('off')  

            plt.tight_layout()

            if op_alpha==False:
                fig.savefig(self.savedir+self.fol+"vb_{}_{}_slice_{}.png".format(op1, op2, self.angle))
            else:
                fig.savefig(self.savedir+self.fol+"vb_{}_{}_alphan{}_b{}_slice_{}.png".format(op1, op2, n,b, self.angle))

            # plt.show()
            
            # if "whole" in self.vb_pre_file:
            # os.makedirs(self.savedir + "/vb_fig/{}/".format(self.calc_fol))
            # tifffile.imwrite(self.savedir +self.fol +  "{}_{}_img.tif".format(op1, op2), img_vx)
        
        return img_vx


if __name__ == '__main__':
    control_points_mod_final = [
    (0.00, (1.00, 0.45, 0.15)),  # orange
    (0.25, (1.00, 1.00, 0.40)),  # yellowish
    (0.50, (0.60, 0.90, 0.90)),  # light cyan
    (0.75, (0.15, 0.75, 1.00)),  # cyan-blue
    (1.00, (0.20, 0.05, 0.85))   # blue-violet
]


    cmap_icefire = LinearSegmentedColormap.from_list(
        "erdc_icefire_darkred",
        control_points_mod_final,
        N=256
    
    )
    ca=cfospy.analysis.read_atlas_data(rdir, vx)
    print("all ID nums", len(ca.ID_all))
    uni_IDs,rev_IDs = ca.get_uni_rIDs()
    # df_sum = ca.get_sum_temp(uni_IDs)
    # print("all region_IDs", len(uni_IDs))

    atlas_mask = ca.get_atlas_mask()
    print(ca.x_num)
    print(ca.y_num)
    print(ca.z_num)
    target_file = "CUBIC-R_Atlas.csv"


    if l_ID !=None: 
        # jtkdir = "/home/gpu_data/data1/JTK_results/"
        
        # CT_df = pd.read_csv(jtkdir+res)
    
        # CT_df = CT_df.drop(["Unnamed: 0"], axis=1)
        # th = 0.1
        # rIDs, ratio= ca.region_periodic(CT_df, l_ID, th)

        ch_IDs, ch_rs, m_IDs, m_rs = ca.get_child_IDs2(l_ID)   #549   313 623
        rIDs = ch_IDs + m_IDs +  [l_ID] 
    
#     th = 0.1 
#     region_per_IDs, ratio = ca.region_periodic(CT_df, l_ID, th)  #1097 549  1065,313, 1089, 623, 698, 512
#                                                               #vx20  #1097  549, 313, 623  315, 512
    # region_per_IDs_all += region_per_IDs
        # rIDs = rIDs[1:]


    # df_atlas_cell = pd.read_csv(rdir + target_file)
    # df_iD = pd.read_csv(rdir + sum_file)
    # print(df_atlas_cell.iloc[0:2,:])
    # atlas_ID_li = df_atlas_cell["atlasID"].tolist()    
    # del df_atlas_cell
    del atlas_mask
    gc.collect()

    xc = int(ca.x_num/2)
    xcoff=5.0*20/vx
    zoff = 25*20/vx
    yoff=10*20/vx
    xoff=-10*20/vx

    for i, rID in enumerate(rIDs):
    # for i, rID in enumerate(rIDs):
        # if rID != 286:
        #     continue
            
        # if i>0:
        #     continue
        
        region = ca.df_allen[ca.df_allen["ID"]==rID]["acronym"].iloc[0]
        print(region)
        print("rID", rID)
                
        if "/" in region:
            region = region.replace("/", "_")

        if not ca.smallID_q(rID):
            ID_li=[]
            # if rID in atlas_ID_li:
            ID_li.append(rID)

            child_IDs, child_regions, middle_IDs, middle_regions = ca.get_child_IDs2(rID)
            for m_ID in child_IDs + middle_IDs:
                # if m_ID in atlas_ID_li:
                    ID_li.append(m_ID)
        else:
                ID_li=[rID]
                
            
        mask0 = np.isin(np.swapaxes((ca.voxel_ID_order_all).reshape(ca.x_num, ca.y_num, ca.z_num), 0, 2), ID_li)

        v_ind = np.where(mask0)
        nv_ind = np.where(~mask0)
        v_ind_xl = v_ind[2][np.where(v_ind[2]<xc)[0]]
        print(region, "vx_ind left", len(v_ind_xl))
        v_ind_xr = v_ind[2][np.where(v_ind[2]>=xc)[0]]
        print(region, "vx_ind_right", len(v_ind_xr))
        
        if len(v_ind_xl)==0:
            xmin_l_o = 0
            xmax_l_o = 0
        else:
            xmin_l_o =  np.min(v_ind_xl)
            xmax_l_o = np.max(v_ind_xl)
            
        if len(v_ind_xr)==0:
            xmin_r_o = 0
            xmax_r_o = 0
        else:
            xmin_r_o = np.min(v_ind_xr)
            xmax_r_o = np.max(v_ind_xr)
        
        ymin_o = np.min(v_ind[1])
        zmin_o = np.min(v_ind[0])

        
       
        ymax_o = np.max(v_ind[1])
        zmax_o = np.max(v_ind[0])
        
        for k, angle in enumerate(angles):
            print(angle)

            if angle=="hor":
                if rID ==286:
                    zmin = int(8632.242/vx)
                    zmax=int(9992.238/vx -zoff)
                    xmin_l = int(7363.42/vx)
                    xmax_l = int(xc+xcoff)
                    ymin = int(9596.6045/vx)
                    xmin_r = int(xc+xcoff)
                    xmax_r=int(9863.196/vx)
                    ymax =int(12207.079/vx)
                
                else:
                    xmin_l = int(xmin_l_o -(xmax_l_o-xmin_l_o)/r)
                    xmin_r = int(xmin_r_o -(xmax_r_o-xmin_r_o)/r)
                    ymin = int(ymin_o -(ymax_o-ymin_o)/r)
                    zmin =zmin_o#int(zmin -(zmax-zmin)/r)
                    xmax_l = int(xmax_l_o +(xmax_l_o-xmin_l_o)/r)
                    xmax_r = int(xmax_r_o +(xmax_r_o-xmin_r_o)/r)
                    ymax = int(ymax_o +(ymax_o-ymin_o)/r)
                    zmax = zmax_o# int(zmax +(zmax-zmin)/r)
                        
                
                
            elif angle=="cor":
                if rID==286:
                    ymin = int(9596.6045/vx +yoff+15*20/vx)
                    ymax=int(12207.079/vx -yoff-50*20/vx)

                    xmin_l = int(7363.42/vx)
                    xmax_l = int(xc+xcoff)
                    xmin_r = int(xc+xcoff)
                    xmax_r=int(9863.196/vx)
                    zmin =int(8519.214/vx)
                    zmax=int(10133.864/vx)
                else:
                    xmin_l = int(xmin_l_o -(xmax_l_o-xmin_l_o)/r)
                    xmin_r = int(xmin_r_o -(xmax_r_o-xmin_r_o)/r)
                    ymin = ymin_o#int(ymin -(ymax-ymin)/r)
                    zmin = int(zmin_o -(zmax_o-zmin_o)/r)
                    xmax_l = int(xmax_l_o +(xmax_l_o-xmin_l_o)/r)
                    xmax_r = int(xmax_r_o +(xmax_r_o-xmin_r_o)/r)
                    ymax = ymax_o #int(ymax +(ymax-ymin)/r)
                    zmax = int(zmax_o +(zmax_o-zmin_o)/r)
                
            elif angle=="sag":
                if rID==286:
                    xmin_l = int(7906.6484/vx  +xoff)
                    xmax_l = int(xc+xcoff)
                    xmin_r = int(xc+xcoff)
                    xmax_r=int(9027.3530/vx -xoff)
                    ymin = int(9596.6045/vx)
                    ymax =int(12207.079/vx)
                    zmin =int(8519.214/vx)
                    zmax=int(10133.864/vx)
                else:
                    xmin_l = xmin_l_o#int(xmin_l -(xmax_l-xmin_l)/r)
                    xmin_r = xmin_r_o#int(xmin_r -(xmax_r-xmin_r)/r)
                    ymin = int(ymin_o -(ymax_o-ymin_o)/r)
                    zmin = int(zmin_o -(zmax_o-zmin_o)/r)
                    xmax_l = xmax_l_o#int(xmax_l +(xmax_l-xmin_l)/r)
                    xmax_r = xmax_r_o#int(xmax_r +(xmax_r-xmin_r)/r)
                    ymax = int(ymax_o +(ymax_o-ymin_o)/r)
                    zmax = int(zmax_o +(zmax_o-zmin_o)/r)
                
                
                
            if xmax_l-xmin_l < xmax_r-xmin_r:
                xmin_r = xmax_r-xmax_l +xmin_l
            elif xmax_l-xmin_l > xmax_r-xmin_r:
                xmax_l = xmax_r - xmin_r  +xmin_l

            if xmin_l < 0 :
                xmin_l =0
            
            if xmin_l>xc and rID!=286:
                xmin_l = xc
            if xmax_r > ca.x_num:
                xmax_r = ca.x_num
            if xmin_r<xc and rID!=286:
                xmin_r = xc
            if ymin < 0 :
                ymin =0
            if ymax > ca.y_num:
                ymax = ca.y_num
            if zmin < 0 :
                zmin =0
            if zmax > ca.z_num:
                zmax = ca.z_num


            #sch
            if rID == 286:
                if xmax_l-xmin_l < xmax_r-xmin_r:
                    xmax_r = xmin_r+xmax_l -xmin_l
                elif xmax_l-xmin_l > xmax_r-xmin_r:
                    xmin_l = -xmax_r + xmin_r  +xmax_l
        
            # 
            

            # #left
            lr = "left"
            try:
                if len(v_ind_xl)!=0:
                    

                    vb_l = calc_vb_phase(rdir, savedir,calc_fol_l, rID, region, vx, r, vb_r, angle, xmin_l, xmax_l, ymin, ymax, zmin, zmax)
                    # print(vb_l.calc_fol)
                    # if not os.path.exists(vb_l.savedir+vb_l.fol+"vb_{}_{}_boder_slice_{}2.png".format("fdr", "nuc_atlas",  angle)):
                    
                    if not vb_l.mox ==0 and not vb_l.moy==0 and not vb_l.moz==0: 
                        vb_l.get_vb_ID()
                        
                        # print("x_b_num: " ,vb_l.x_b_num)
                        # print("y_b_num: " ,vb_l.y_b_num)
                        # print("z_b_num: " ,vb_l.z_b_num)
                    
                        if vb_l.x_b_num < blockdim_x:
                            blockdim_x = 1
                        if vb_l.y_b_num < blockdim_y:
                            blockdim_y = 1
                        if vb_l.z_b_num < blockdim_z:
                            blockdim_z = 1
                        if not os.path.exists(vb_l.savedir+ "2nd/"+vb_l.root_fol_count+ "/CT44_06_vb_CT_{}.bin".format("2nd")):
                            args = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format(vb_l.x_b_num, vb_l.y_b_num, vb_l.z_b_num , xmin_l, ymin, zmin, xmax_l, ymax, zmax, region, vx, vb_r, vb_l.mox, vb_l.moy, vb_l.moz, r, angle, vb_l.savedir, vb_l.calc_fol, ants_dir_name_point_file, intensity_file, ncore, blockdim_x,blockdim_y,blockdim_z)
                            if calc_type=="count" or calc_type=="count_ratio":
                                outc = "env CUDA_VISIBLE_DEVICES={} ./vx_pro_c2 ".format(gpu_num) + args
                            else:
                                outc = "env CUDA_VISIBLE_DEVICES={} ./vx_pro_ci ".format(gpu_num) + args
                            print(outc)
                            subprocess.run([outc], shell=True)
                        # if not os.path.exists(vb_l.savedir+vb_l.fol+"cos_1st2nd.csv"):
                        vb_l.make_cos_vb(sample_names)
                        img_vx = vb_l.overlay_images(rID,  size, s, True, "fdr", "nuc_atlas", op_alpha, n, b)
                        vb_l.make_vb_image_edge(rID, img_vx, lr, "fdr", "nuc_atlas")
                    else:
                        print(region, " left, mo is 0")
                    del vb_l
                    gc.collect()
                    # else:
                    #     print("ID{}_{}_{} already exist".format(rID, lr, angle))
                else:
                    print(region, " no voxel in left")
            except:
                traceback.print_exc()
                
            ## right
            lr = "right"
            #hor and cor
            if rID == 286:
                if angle =="hor" or angle =="cor":
                    xmin_r = xmin_r -5
                # xmax_r = xmax_r -5
            try:
                if len(v_ind_xr) !=0:
                    vb_right = calc_vb_phase(rdir, savedir, calc_fol_r, rID, region, vx, r, vb_r, angle, xmin_r, xmax_r, ymin, ymax, zmin, zmax)
                    # print(vb_right.calc_fol)

                    # if not os.path.exists(vb_right.savedir+vb_right.fol+"vb_{}_{}_boder_slice_{}2.png".format("fdr", "nuc_atlas",  angle)):
                    

                    if not vb_right.mox ==0 and not vb_right.moy==0 and not vb_right.moz==0: 
                        vb_right.get_vb_ID()
                        if  vb_right.x_b_num < blockdim_x:
                            blockdim_x = 1
                        if  vb_right.y_b_num < blockdim_y:
                            blockdim_y = 1
                        if  vb_right.z_b_num < blockdim_z:
                            blockdim_z = 1
                        if not os.path.exists(vb_right.savedir+ "2nd/"+vb_right.root_fol_count+ "/CT44_06_vb_CT_{}.bin".format("2nd")):
                            args = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format(vb_right.x_b_num, vb_right.y_b_num, vb_right.z_b_num , xmin_r, ymin, zmin, xmax_r, ymax, zmax, region, vx, vb_r, vb_right.mox, vb_right.moy, vb_right.moz, r, angle, vb_right.savedir, vb_right.calc_fol, ants_dir_name_point_file, intensity_file, ncore, blockdim_x, blockdim_y, blockdim_z)
                            if calc_type=="count" or calc_type=="count_ratio":
                                outc = "env CUDA_VISIBLE_DEVICES={} ./vx_pro_c2 ".format(gpu_num) + args
                            else:
                                outc = "env CUDA_VISIBLE_DEVICES={} ./vx_pro_ci ".format(gpu_num) + args
                            print(outc)
                            subprocess.run([outc], shell=True)
                        # if not os.path.exists(vb_right.savedir+vb_right.fol+"cos_1st2nd.csv"):
                        vb_right.make_cos_vb(sample_names)
                        img_vx =vb_right.overlay_images(rID,  size, s, True, "fdr", "nuc_atlas", op_alpha, n, b)
                        vb_right.make_vb_image_edge(rID, img_vx, lr, "fdr", "nuc_atlas")
                    else:
                        print(region, " right, mo is 0")
                    del vb_right
                    gc.collect()
                    # else:
                    #     print("ID{}_{}_{} already exist".format(rID, lr, angle))
                else:
                    print(region, " no voxel in right")
            except:
                traceback.print_exc()
                
           
                