# cfos_circadian
This repository contains scripts for analyzing the circadian rhythmicity of c-Fos expression in mouse whole-brain samples and generating corresponding figures.

## Requirements
* Python 3.10
* numpy
* scipy
* pandas
* matplotlib
* statsmodels
* multiprocess
* tifffile
* nibabel
* numba
* scikit-image
* scikit-learn
* cupy
* opencv-python
* seaborn
* xgboost
* lightgbm
* ANTs 2.5.0

If you would like to analyze cell counts at the voxel level, please make sure the following GPU-compatible libraries are installed: 
* CUDA Toolkit
* NVIDIA HPC SDK (https://developer.nvidia.com/hpc-sdk)

You need to compile the CUDA source files (`*.cu`) in the `src` directory when performing voxel-level cell counting.

## Usage

### 1. Image normalization (ANTs)
`ANTs_image.ipynb` normalizes nuclear and c-Fos staining images of mouse whole brains using the Neuron atlas.  
The Neuron atlas is available [here](https://drive.google.com/drive/u/1/folders/1klfrOAqJ7sOvPBMb1-6MniIjWF8LsiHU).

### 2. AI-based cell detection
`deconvolution_and_peak_detection.ipynb` and `AI-based_CellDetection.ipynb` perform deconvolution of c-Fos staining images and detect candidate c-Fos–positive cells using machine learning.  
A GPU is required to compute distances between the point spread function (PSF) and detected candidates.

### 3. Rhythmicity analysis of regional c-Fos–positive cells
`ANTs_points_cosinor_test_ai_DD.ipynb` analyzes the rhythmicity of regional c-Fos–positive cells using the analytic cosinor test.  
The original source code for the analytic cosinor test is available [here](https://github.com/OrganismalSystemsBiology/analytic_cosinor).

### 4. Voxel-wise rhythmicity analysis
`cfos_voxel-wise_analysis_gpu_whole_img.py` analyzes voxel-wise rhythmicity of c-Fos–positive cells across the whole brain.  
2D slices, 2D projections, and 3D rotational visualizations by region are generated using:  
`cfos_whole_brain_voxel_slice.ipynb`  
`cfos_voxel-wise_projection_gpu.py`  
`cfos_voxel_phase_3D_rotation.ipynb`  
  
Tiff files for whole-brain voxel-wise analysis are available [here](https://drive.google.com/drive/u/1/folders/1klfrOAqJ7sOvPBMb1-6MniIjWF8LsiHU).

### 5. Time prediction using the timetable method
`cfos_time_prediction_timetable_method.ipynb` predicts the internal time of a brain sample using the timetable method.  
For details, see [Ueda et al., PNAS (2004)](https://www.pnas.org/doi/10.1073/pnas.0401882101).

### 6. Time prediction using CYCLOPS 2.0
`cfos_time_prediction_CYCLOPS.ipynb` predicts the internal time of a brain sample using CYCLOPS 2.0.  
For details, see [Anafi et al., PNAS (2017)](https://www.pnas.org/doi/10.1073/pnas.1619320114).

## Contact
K. Yamashita — kyamashi@m.u-tokyo.ac.jp  
F. L. Kinoshita — kinoshita.lee@gmail.com

If you use this code in your research, please cite the following original paper:  
Yamashita K. *et al.* (under review)  
**A whole-brain single-cell atlas of circadian neural activity in mice.**
