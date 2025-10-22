# cfos_circadian
This repository contains scripts for analyzing the circadian rhythmicity of c-Fos expression in mouse whole-brain samples and generating corresponding figures.

## Requirements
* Python 3.10
* numpy
* scipy
* pandas
* statsmodels
* tifffile
* nibabel
* scikit-image
* opencv-python
* matplotlib
* seaborn
* numba
* multiprocess
* cupy
* scikit-learn
* xgboost
* lightgbm
* ANTs 2.5.0

If you would like to analyze cell counts at the voxel level, please make sure the following GPU-compatible libraries are installed: 
* CUDA Toolkit 12.0
* NVIDIA HPC SDK (https://developer.nvidia.com/hpc-sdk)

You need to compile the CUDA source files (`*.cu`) in the `src` directory when performing voxel-level cell counting.

## Usage

### 1. AI-based cell detection
`deconvolution.ipynb` and `AI_based_cell_detection.ipynb` perform deconvolution of c-Fos staining images and detect c-Fos–positive cells using machine learning.  
A GPU is required to compute distances between the point spread function (PSF) and detected candidates.

### 2. Image normalization (ANTs)
`ANTs_registration.ipynb` normalizes nuclear and c-Fos staining images of whole mouse brains using the Neuron atlas.  
The Neuron atlas is available [here](https://drive.google.com/drive/u/1/folders/1klfrOAqJ7sOvPBMb1-6MniIjWF8LsiHU).

### 3. Rhythmicity analysis of regional c-Fos–positive cells
`cell_region_summary.ipynb` and `cosinor_test.ipynb` analyze the rhythmicity of regional c-Fos–positive cells using the analytic cosinor test.  
The original source code for the analytic cosinor test is available [here](https://github.com/OrganismalSystemsBiology/analytic_cosinor).

### 4. Voxel-wise rhythmicity analysis
`voxel_analysis_whole_GPU.py` analyzes voxel-wise rhythmicity of c-Fos–positive cells across the whole brain.  
2D slices, 2D projections, and 3D visualizations by region are generated using:  
- `voxel_analysis_slice.ipynb`  
- `voxel_analysis_projection_GPU.py`  
- `voxel_analysis_3D_view.ipynb`  
  
Files for whole-brain voxel-wise analysis are available [here](https://drive.google.com/drive/u/1/folders/1klfrOAqJ7sOvPBMb1-6MniIjWF8LsiHU), including Tiff image files for rhythmicity analysis of user-defined regions.

### 5. Time prediction using the timetable method
`time_prediction_timetable.ipynb` predicts the internal time of single brain samples using the timetable method.  
For details, see [Ueda et al., PNAS (2004)](https://www.pnas.org/doi/10.1073/pnas.0401882101).

### 6. Time prediction using CYCLOPS 2.0
`cyclops.py` and `time_prediction_CYCLOPS.ipynb` predict the internal time of single brain samples using CYCLOPS 2.0.  
For details, see [Anafi et al., PNAS (2017)](https://www.pnas.org/doi/10.1073/pnas.1619320114).

## Contact
K. Yamashita — kyamashi@m.u-tokyo.ac.jp  
F. L. Kinoshita — kinoshita.lee@gmail.com

If you use this code in your research, please cite the following original paper:  
Yamashita K. *et al.* (under review)  
**A whole-brain single-cell atlas of circadian neural activity in mice.**
