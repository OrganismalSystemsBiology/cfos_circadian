import os
import gc
import concurrent.futures
import cv2
import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import maximum_filter, gaussian_filter
import matplotlib.pyplot as plt
from numba import njit
from numba.typed import List

def Gaussian_filter_CPU(img, sigma):
    dst = gaussian_filter(img, sigma=sigma, mode="reflect")
    return dst


def resize3D(img, new_size, current_size):
    ratio = new_size / current_size
    img = img.astype(np.float32)
    img_tmp = []
    for img0 in img:
        img1 = cv2.resize(img0, dsize=None, fx=ratio, fy=ratio, interpolation =cv2.INTER_CUBIC)
        img_tmp.append(img1)
    img_tmp = np.array(img_tmp)
    img_tmp =img_tmp.T

    img_tmp2 = []
    for img0 in img_tmp:
        img1 = cv2.resize(img0, dsize=None, fx=ratio, fy=1, interpolation = cv2.INTER_CUBIC)
        img_tmp2.append(img1)
    img_tmp2 = np.array(img_tmp2)
    return img_tmp2.T   


def min_max(x: np.ndarray):
    minval = x.min(axis=None, keepdims=True)
    maxval = x.max(axis=None, keepdims=True)
    result = (x - minval) / (maxval - minval + 1e-12)
    return result


def crop_shift_resize(orig_cube, new_size, crop_size, orig_size):        # Crop the cube, Shift the model by one pixel, and Resize it
    start = (orig_size - crop_size)//2
    end = start + crop_size
    crop_cube = orig_cube[start:end, start:end, start:end]
    
    cube_list = []
    for j in range(-1, 2):
        for k in range(-1, 2):
            shift_cube = np.roll(crop_cube, j, axis=1)
            shift_cube = np.roll(shift_cube, k, axis=2)
            new_cube = resize3D(shift_cube, new_size, crop_size)
            cube_list.append(new_cube)
    return cube_list    


def make_PSF(sigma_values, new_size, orig_size):
    all_psfs = []
    for sigma in sigma_values:
        psf_list = []
        orig_cube = np.zeros((orig_size, orig_size, orig_size))
        orig_cube[orig_size//2, orig_size//2, orig_size//2] = 1

        gaus_psf = Gaussian_filter_CPU(orig_cube, sigma=sigma)
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    g = gaus_psf.copy()
                    shift_cube = np.roll(g, i, axis=0)
                    shift_cube = np.roll(shift_cube, j, axis=1)
                    shift_cube = np.roll(shift_cube, k, axis=2)
                    new_cube = resize3D(shift_cube, new_size, orig_size)
                    psf_list.append(new_cube)
        
        norm_psf_list = [min_max(p) for p in psf_list]
        all_psfs.extend(norm_psf_list)

    print(f'Number of PSF models: {len(all_psfs)}')
    psf_data = np.array(all_psfs)
    return psf_data


def makemask(size):
    half = size//2
    mask = np.zeros((size, size, size))
    for i in range(-half, half+1):
        for j in range(-half, half+1):
            for k in range(-half, half+1):
                if((abs(i)+abs(j)+abs(k)) < half+1):
                    mask[half+i, half+j, half+k] = 1
    return mask


def int_convert(x:np.float32):
    if x<0:
        x=int(x-0.5)
    else:
        x=int(x+0.5)
    return x


def make_line_3d(sigma_values, new_size, crop_size, orig_size,
                 start1=-90, end1=90, step1=15,    # xy angle (-90~89)
                 start2=0, end2=61, step2=15):    # z angle (0~90)
    
    half = orig_size//2
    orig_cube = np.zeros((orig_size, orig_size, orig_size), dtype=np.float32)
    line_3d_list = []
    thetas1 = np.arange(start1, end1, step1)    # xy angle
    thetas2 = np.arange(start2, end2, step2)    # z angle
    
    for th2 in thetas2:
        for th1 in thetas1:
            tan1 = np.tan(np.radians(th1))
            cos1 = np.cos(np.radians(th1))
            sin1 = np.sin(np.radians(th1))
            tan2 = np.tan(np.radians(th2))

            if cos1!=0:
                x = half
                y = x*tan1
                z = x/cos1 * tan2
                if abs(y)<=half and abs(z)<=half:
                    for i in range(-half, half+1):
                        x = i
                        y = x*tan1
                        z = x/cos1 * tan2
                        y = int_convert(y)
                        z = int_convert(z)
                        orig_cube[z+half, y+half, x+half] = 1
                    line_3d_list.append(orig_cube)
                    orig_cube = np.zeros((orig_size, orig_size, orig_size))
                    continue

            if tan1!=0:
                y = half
                x = y/tan1
                z = y/sin1 * tan2
                if abs(x)<=half and abs(z)<=half:
                    for i in range(-half, half+1):
                        y = i
                        x = y/tan1
                        z = y/sin1 * tan2
                        x = int_convert(x)
                        z = int_convert(z)
                        orig_cube[z+half, y+half, x+half] = 1
                    line_3d_list.append(orig_cube)
                    orig_cube = np.zeros((orig_size, orig_size, orig_size))
                    continue

            if tan2!=0:
                z = half
                x = z/tan2*cos1
                y = x*tan1
                if abs(x)<=half and abs(y)<=half:
                    for i in range(-half, half+1):
                        z = i
                        x = z/tan2*cos1
                        y = x*tan1
                        x = int_convert(x)
                        y = int_convert(y)
                        orig_cube[z+half, y+half, x+half] = 1
                    line_3d_list.append(orig_cube)
                    orig_cube = np.zeros((orig_size, orig_size, orig_size))

    norm_line_3d_list = []
    for line_3d in line_3d_list:
        for sigma in sigma_values:
            gaus_line = Gaussian_filter_CPU(line_3d, sigma=sigma)
            line_list = crop_shift_resize(gaus_line, new_size, crop_size, orig_size)
            
            for line in line_list:
                norm_line = min_max(line)
                norm_line_3d_list.append(norm_line)
    
    print(f'Number of 3D lines: {len(norm_line_3d_list)}')
    line_data = np.array(norm_line_3d_list)
    return line_data


def read_tiff_stack(folder_path, start_index, num_images):
    # Get the list of TIFF files
    print(f"Load tiff: {start_index}- {num_images}files")
    tiff_files = [os.path.join(folder_path, filename)
              for filename in os.listdir(folder_path)
              if filename.lower().endswith(('.tif', '.tiff'))] 
    # Sort the files to ensure correct order
    tiff_files.sort()
    # Select the specified range of files
    tiff_files = tiff_files[start_index:(start_index + num_images)]
    first_tiff = tifffile.imread(tiff_files[0])
    shape = (len(tiff_files),) + first_tiff.shape
    stack = np.empty(shape, dtype=np.uint16)
    # Read each TIFF file and store it in the stack
    for i, tiff_file in enumerate(tiff_files):
        stack[i] = tifffile.imread(tiff_file)
    return stack


def Peakdetection_CPU(input_data, kernel_size, params):
    dst = maximum_filter(input_data, size=kernel_size, mode='reflect', cval=0.0, origin=0)
    dst[input_data != dst] = 0
    dst[input_data < params[0]] = 0
    return dst


@njit
def peaks_to_result(img, indices, size):
    z_idx = indices[0]
    y_idx = indices[1]
    x_idx = indices[2]

    half = size // 2
    D, H, W = img.shape

    n = z_idx.shape[0]
    result = np.empty((n, 5), np.uint16)
    count = 0

    for idx in range(n):
        z = int(z_idx[idx])
        y = int(y_idx[idx])
        x = int(x_idx[idx])

        if (x < half + 1) or (x > W - half - 1):
            continue
        if (y < half + 1) or (y > H - half - 1):
            continue
        if (z < half + 1) or (z > D - half - 1):
            continue

        centval = np.uint16(img[z, y, x])
        minval = np.uint16(65535)

        for i in range(-half, half + 1):
            zi = z + i
            for j in range(-half, half + 1):
                yj = y + j
                for k in range(-half, half + 1):
                    xk = x + k
                    curval = np.uint16(img[zi, yj, xk])
                    if minval > curval:
                        minval = curval

        deltaI = np.uint16(int(centval) - int(minval))

        result[count, 0] = np.uint16(x)
        result[count, 1] = np.uint16(y)
        result[count, 2] = np.uint16(z)
        result[count, 3] = centval
        result[count, 4] = deltaI
        count += 1

    return result[:count]


@njit
def CalcSimilarity_PSF(img: np.ndarray,\
                    X: np.ndarray,\
                    Y: np.ndarray,\
                    Z: np.ndarray,\
                    psfs: np.ndarray,\
                    Delta: np.float32,\
                    min_differ: np.float32,\
                    psf_size: np.int16):
    
    half = psf_size//2
    listresult = []
    for n in range(len(X)):
        x = X[n]
        y = Y[n]
        z = Z[n]
        if(x < half+1) or (x > img.shape[2]-half-1):
            continue
        if(y < half+1) or (y > img.shape[1]-half-1):
            continue
        if(z < half+1) or (z > img.shape[0]-half-1):
            continue
            
        centval = np.float32(img[z, y, x])
        minval = np.float32(65535)
        for i in range(-half, half+1):
            for j in range(-half, half+1):
                for k in range(-half, half+1):
                    curval = np.float32(img[z+i, y+j, x+k])
                    if(minval > curval):
                        minval = curval

        deltaI = np.float32(centval - minval)

        differ = np.float32(100000)
        if(delta_val > Delta):
            for p in range(len(psfs)): 
                d = np.float32(0)
                for i in range(-half, half+1):
                    for j in range(-half, half+1):
                        for k in range(-half, half+1):
                            if((abs(i)+abs(j)+abs(k)) < half+1):
                                d += abs((np.float32(img[z+i,y+j,x+k])-minval)/delta_val - np.float32(psfs[p][half+i, half+j, half+k])) #L1
                if(differ > d):
                    differ = d
            #print(str(min_differ)+":"+str(differ))
            if(differ < min_differ):
                listresult.append((x, y, z))  
                
    return listresult


def process_CalcSimilarity_PSF(stack, indices, threads, CalcSimilarity_PSF, psfs, params, psf_size):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        future_to_segment = {}
        point_num = len(indices[0])
        segment_size = point_num // threads + 1
        for i in range(threads):
            start_idx = i*segment_size
            if i < (threads-1):
                end_idx = (i+1)*segment_size
            else:
                end_idx = point_num
            future = executor.submit(CalcSimilarity_PSF, stack, 
                                     indices[2][start_idx:end_idx], 
                                     indices[1][start_idx:end_idx], 
                                     indices[0][start_idx:end_idx], 
                                     psfs, params[1], params[2], psf_size)
            future_to_segment[future] = i

        for future in concurrent.futures.as_completed(future_to_segment):
            segment_result = future.result()
            results.extend(segment_result)
    
    return results
    

@njit
def CalcSimilarity_line(img: np.ndarray,\
                   X: np.ndarray,\
                   Y: np.ndarray,\
                   Z: np.ndarray,\
                   lines: np.ndarray,\
                   max_differ: np.float32,\
                   line_size: np.int16):
    
    half = line_size//2
    listresult = []
    for n in range(len(X)):
        x = X[n]
        y = Y[n]
        z = Z[n]
        if(x < half+1) or (x > img.shape[2]-half-1):
            continue
        if(y < half+1) or (y > img.shape[1]-half-1):
            continue
        if(z < half+1) or (z > img.shape[0]-half-1):
            continue
            
        centval=np.float32(img[z, y, x])
        minval=np.float32(65535)
        for i in range(-half, half+1):
            for j in range(-half, half+1):
                for k in range(-half, half+1):
                    curval = np.float32(img[z+i,y+j,x+k])
                    if(minval > curval):
                        minval = curval

        deltaI = np.float32(centval - minval)
 
        differ = np.float32(100000)
        for l in range(len(lines)): 
            d = np.float32(0)
            for i in range(-half, half+1):
                for j in range(-half, half+1):
                    for k in range(-half, half+1):
                        curval = np.float32(img[z+i, y+j, x+k])
                        diff = np.float32(lines[l][half+i, half+j, half+k]) - ((curval-minval)/delta_val)
                        if diff > 0:
                            d += diff
            if(differ > d):
                differ = d
        if(differ > max_differ):
            differ = np.round(differ, 2)
            listresult.append((x, y, z, centval, delta_val, differ))  

    return listresult
                       

def process_CalcSimilarity_line(stack, indices, threads, CalcSimilarity_line, lines, params, line_size):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        future_to_segment = {}
        point_num = len(indices[0])
        segment_size = point_num // threads + 1
        for i in range(threads):
            start_idx = i * segment_size
            if i < (threads-1):
                end_idx = (i+1)*segment_size
            else:
                end_idx = point_num
            future = executor.submit(CalcSimilarity_line, stack, 
                                     indices[2][start_idx:end_idx], 
                                     indices[1][start_idx:end_idx], 
                                     indices[0][start_idx:end_idx], 
                                     lines, params[3], line_size)
            future_to_segment[future] = i

        for future in concurrent.futures.as_completed(future_to_segment):
            segment_result = future.result()
            results.extend(segment_result)
    
    return results
    

def makepointimage(coordinates:list, src:np.ndarray, sigma, makedistance=False):
    dst = np.zeros(src.shape, dtype=np.uint16)
    if len(coordinates) > 0:
        coordinates = List(coordinates)
        makepointimage_nb(coordinates, dst, makedistance)
    if makedistance == False:
        return(Gaussian_filter_CPU(dst, sigma))
    return dst
    
    
@njit
def makepointimage_nb(src:List, dst:np.ndarray, makedistance:bool):
    if makedistance:
        for i in range(len(src)):
            if src[i][4] < 10:
                dst[src[i][2]][src[i][1]][src[i][0]] = (10 - src[i][4]) * 100
    else:
        for i in range(len(src)):
            dst[src[i][2]][src[i][1]][src[i][0]] = 255


def GetPeaks(FP, FPw, params, peak_size, psf_size, batchsize, overlap, threads, save_points=True):
    """Detect peaks only and export coordinates and point images."""
    os.makedirs(FPw, exist_ok=True)
    
    points_dir = None
    if save_points:
        points_dir = os.path.join(FPw, "points")
        os.makedirs(points_dir, exist_ok=True)

    column_names = ["X", "Y", "Z", "intensity", "deltaI"]
    df = pd.DataFrame(columns=column_names)
    
    slice_num = len(os.listdir(FP))
    if slice_num % batchsize == 0:
        batch_num = slice_num // batchsize
    else:
        batch_num = slice_num // batchsize + 1

    for i in range(batch_num):
        print(f"Batch {i}/{batch_num}")
        
        if i == 0:
            startz = 0
            tiff_stack = read_tiff_stack(FP, startz, batchsize + overlap)    # Set overlap at the end of the first stack
        elif i == batch_num - 1:
            startz = i * batchsize - overlap
            tiff_stack = read_tiff_stack(FP, startz, batchsize + overlap)    # Set overlap at the beginning of the last stack
        else:
            startz = i * batchsize - overlap
            tiff_stack = read_tiff_stack(FP, startz, batchsize + (2 * overlap))    # Set overlap at both ends of the other stacks
        
        tiff_stack = np.where(np.isnan(tiff_stack), 0, tiff_stack)   # Replace NaN with 0
        shape = tiff_stack.shape
        print(f"tiff stack shape: {shape}")

        # Peak detection (CPU)
        peaks = Peakdetection_CPU(tiff_stack, peak_size, params)
        peak_indices = np.nonzero(peaks)
        print(f"Peaks: {len(peak_indices[0])}")
        del peaks

        result_data = peaks_to_result(tiff_stack, peak_indices, psf_size)

        df_batch = pd.DataFrame(result_data, columns=column_names).dropna(axis=1, how="all")

        if len(df_batch) > 0:
            if "Z" in df_batch.columns:
                if i == 0:
                    df_batch = df_batch[df_batch["Z"] < batchsize]
                elif i == batch_num - 1:
                    df_batch = df_batch[df_batch["Z"] > overlap - 1]
                else:
                    df_batch = df_batch[(df_batch["Z"] > overlap - 1) & (df_batch["Z"] < batchsize + overlap)]
                df_batch.loc[:, "Z"] = df_batch["Z"] + startz

            # Export point images (avoid duplicate overlaps)
            if save_points:
                points = makepointimage(result_data, tiff_stack, 1)
                    
                count = 0
                if i == batch_num - 1:
                    for z in range(shape[0]):
                        filename = os.path.join(points_dir, f"{i*batchsize - overlap + z:08}.tif")
                        if not os.path.exists(filename):
                            tifffile.imwrite(filename, points[z])
                            count += 1
                else:
                    for z in range(points.shape[0] - overlap):
                        if i == 0:
                            filename = os.path.join(points_dir, f"{i*batchsize + z:08}.tif")
                        else:
                            filename = os.path.join(points_dir, f"{i*batchsize - overlap + z:08}.tif")
                        if not os.path.exists(filename):
                            tifffile.imwrite(filename, points[z])
                            count += 1
                print(f"Saved {count} slices\n")

            del tiff_stack
            df = pd.concat([df, df_batch], ignore_index=True)
            
        else:
            # No peaks: write zero images to keep slice continuity
            if save_points:
                zeros_file = np.zeros((shape[1], shape[2]), dtype=np.uint16)
                count = 0
                for z in range(shape[0]):
                    if i == 0:
                        filename = os.path.join(points_dir, f"{i*batchsize + z:08}.tif")
                    else:
                        filename = os.path.join(points_dir, f"{i*batchsize - overlap + z:08}.tif")
                    if not os.path.exists(filename):
                        tifffile.imwrite(filename, zeros_file)
                        count += 1
                print(f"Saved {count} slices\n")
            
    duplicates = df.duplicated(keep=False)
    duplicate_count = duplicates.sum()
    print(f'Duplicate : {duplicate_count}')
    
    out_csv = os.path.join(FPw, "coordinates.csv")
    df.to_csv(out_csv, index=False)
    print(f"Detected Cells in total: {len(df)}\n")
    return df
    

def GetCells(FP, FPw, params, peak_size, psf_sizes, psf_sigmas, line_sizes, line_3d_sigmas, batchsize, overlap, threads):
    """Detect cells: peak → PSF similarity filter → line exclusion; export coords and point images."""
    os.makedirs(FPw, exist_ok=True)
    points_dir = os.path.join(FPw, "points")
    os.makedirs(points_dir, exist_ok=True)

    psfs = make_PSF(psf_sigmas, psf_sizes[0], psf_sizes[1])
    mask = makemask(psf_sizes[0])
    psfs = psfs * mask
    lines = make_line_3d(line_3d_sigmas, line_sizes[0], line_sizes[1], line_sizes[2])

    column_names = ["X", "Y", "Z", "intensity", "deltaI", "dissimilarity"]
    df = pd.DataFrame(columns=column_names)
    
    slice_num = len(os.listdir(FP))
    if slice_num % batchsize == 0:
        batch_num = slice_num // batchsize
    else:
        batch_num = slice_num // batchsize + 1

    for i in range(batch_num):
        print(f"Batch {i}/{batch_num}")
        
        if i == 0:
            startz = 0
            tiff_stack = read_tiff_stack(FP, startz, batchsize + overlap)    # Set overlap at the end of the first stack
        elif i == batch_num - 1:
            startz = i * batchsize - overlap
            tiff_stack = read_tiff_stack(FP, startz, batchsize + overlap)    # Set overlap at the beginning of the last stack
        else:
            startz = i * batchsize - overlap
            tiff_stack = read_tiff_stack(FP, startz, batchsize + (2 * overlap))    # Set overlap at both ends of the other stacks
        
        tiff_stack = np.where(np.isnan(tiff_stack), 0, tiff_stack)   # Replace NaN with 0
        shape = tiff_stack.shape
        print(f"tiff stack shape: {shape}")

        # Peak detection (CPU)
        peaks = Peakdetection_CPU(tiff_stack, peak_size, params)
        peak_indices = np.nonzero(peaks)
        print(f"Peaks: {len(peak_indices[0])}")
        del peaks
        
        # PSF similarity filtering
        result_psf = process_CalcSimilarity_PSF(tiff_stack, peak_indices, threads, CalcSimilarity_PSF, psfs, params, psf_sizes[0])
        print(f"Detected cells similar to PSF templates: {len(result_psf)}")
        del peak_indices
        
        # Convert (x, y, z) tuple list to (z, y, x) NumPy array tuple
        x_indices = np.array([r[0] for r in result_psf], dtype=np.uint16)    
        y_indices = np.array([r[1] for r in result_psf], dtype=np.uint16)
        z_indices = np.array([r[2] for r in result_psf], dtype=np.uint16)
        psf_indices = (z_indices, y_indices, x_indices)
        del result_psf

        # Line exclusion (3D line templates)
        result_data = process_CalcSimilarity_line(tiff_stack, psf_indices, threads, CalcSimilarity_line, lines, params, line_sizes[0])
        print(f"Detected cells after line exclusion: {len(result_data)}")
        del psf_indices
        
        df_batch = pd.DataFrame(result_data, columns=column_names).dropna(axis=1, how="all")

        if len(df_batch) > 0:
            if "Z" in df_batch.columns:
                if i == 0:
                    df_batch = df_batch[df_batch["Z"] < batchsize]
                elif i == batch_num - 1:
                    df_batch = df_batch[df_batch["Z"] > overlap - 1]
                else:
                    df_batch = df_batch[(df_batch["Z"] > overlap - 1) & (df_batch["Z"] < batchsize + overlap)]
                df_batch.loc[:, "Z"] = df_batch["Z"] + startz

            # Export point images (avoid duplicate overlaps)
            points = makepointimage(result_data, tiff_stack, 1)
                
            count = 0
            if i == batch_num - 1:
                for z in range(shape[0]):
                    filename = os.path.join(points_dir, f"{i*batchsize - overlap + z:08}.tif")
                    if not os.path.exists(filename):
                        tifffile.imwrite(filename, points[z])
                        count += 1
            else:
                for z in range(points.shape[0] - overlap):
                    if i == 0:
                        filename = os.path.join(points_dir, f"{i*batchsize + z:08}.tif")
                    else:
                        filename = os.path.join(points_dir, f"{i*batchsize - overlap + z:08}.tif")
                    if not os.path.exists(filename):
                        tifffile.imwrite(filename, points[z])
                        count += 1
            print(f"Saved {count} slices\n")

            del tiff_stack
            df = pd.concat([df, df_batch], ignore_index=True)
            
        else:
            # No peaks: write zero images to keep slice continuity
            zeros_file = np.zeros((shape[1], shape[2]), dtype=np.uint16)
            count = 0
            for z in range(shape[0]):
                if i == 0:
                    filename = os.path.join(points_dir, f"{i*batchsize + z:08}.tif")
                else:
                    filename = os.path.join(points_dir, f"{i*batchsize - overlap + z:08}.tif")
                if not os.path.exists(filename):
                    tifffile.imwrite(filename, zeros_file)
                    count += 1
            print(f"Saved {count} slices\n")
            
    duplicates = df.duplicated(keep=False)
    duplicate_count = duplicates.sum()
    print(f'Duplicate : {duplicate_count}')
    
    out_csv = os.path.join(FPw, "coordinates.csv")
    df.to_csv(out_csv, index=False)
    print(f"Detected Cells in total: {len(df)}\n")
    return df


def MakeStructureImage(FP, FPw, res, antssize):
    os.makedirs(FPw, exist_ok=True)
    print(FP)
    tiff_stack = read_tiff_stack(FP, 0, len(os.listdir(FP)))
    tifffile.imwrite(FPw+"/Structure_image.tif",resize3D(tiff_stack,res,antssize))