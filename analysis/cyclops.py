import numpy as np
import os
import subprocess
import datetime, re

#python3 cyclops.py

src = "/home/ubuntu/my_data/"
j_dir = src + "data8/cfos_app/timetable/CYCLOPS-2.0-main/"
jl_f = "CYCLOPS_2_0_Template.jl"

min_regions = [20]#np.arange(100,651,50)#[525, 425, 475, 375, 325]
eigen_maxs = [9]#np.arange(4,10,1)
model_ns = [80]
min_cvs = [0.15]#np.arange(0.05, 0.55, 0.1)#[0.2]#[0.05, 0.1, 0.2, 0.3]
max_cvs = [0.8]#np.arange(0.5, 1.0, 0.1)#[0.6]#[0.7, 0.8, 0.9, 1.0]
core = 20
#2024-06-18T01_50_00_eigen_max_13_seed_max_CV_0_5_seed_min_CV_0_05_seed_mth_Gene_360

results_pre_fol= j_dir + "results/" 

min_region_d = 3
eigen_max_d = "99"
mincv_d = "0_85_"
maxcv_d = "0_1_"
# # 基準となる日時
# reference_date_str = "2024-06-17T15_00_00"
# reference_date = datetime.datetime.strptime(reference_date_str, "%Y-%m-%dT%H_%M_%S")

files = os.listdir(results_pre_fol)


param_sets = []


for i, text in enumerate(files):
#     if i>0:
#         continue
        
#     try:
#         date_str = ("_").join(text.split('_')[0:3])  # 日付部分を抽出
# #         print(date_str)
#         date_ex = datetime.datetime.strptime(date_str, "%Y-%m-%dT%H_%M_%S")
#     except ValueError:
#         traceback.print_exc()
    
    # if date_ex >= reference_date:
        
        # seed_min_CV_の後ろの数字を抽出
        seed_min_cv_match = re.search(r'seed_min_CV_([\d_]+)', text)
        if seed_min_cv_match:
            seed_min_cv_value = seed_min_cv_match.group(1)
        else:
            seed_min_cv_value = mincv_d

        # eigen_max_の後ろの数字を抽出
        eigen_max_match = re.search(r'eigen_max_(\d+)', text)
        if eigen_max_match:
            eigen_max_value = eigen_max_match.group(1)
        else:
            eigen_max_value =  eigen_max_d

        # seed_max_CV_の後ろの数字を抽出
        seed_max_cv_match = re.search(r'seed_max_CV_([\d_]+)', text)
        if seed_max_cv_match:
            seed_max_cv_value = seed_max_cv_match.group(1)
        else:
            seed_max_cv_value = maxcv_d
            
        # seed_mth_Gene_の後ろの数字を抽出
        seed_mth_Gene_match = re.search(r'seed_mth_Gene_([\d_]+)', text)
        if seed_mth_Gene_match:
            seed_mth_Gene_value = seed_mth_Gene_match.group(1)
        else:
            seed_mth_Gene_value = mincv_d

        # 結果を出力
        # print(f'eigen_max_value: {eigen_max_value}')
        # print(f'seed_max_cv_value: {seed_max_cv_value}')
        # print(f'seed_min_cv_value: {seed_min_cv_value}')
        # print(f'seed_mth_Gene_value: {seed_mth_Gene_value}')
        
        param_sets.append([int(eigen_max_value), round(float(seed_min_cv_value.replace('_', '.').rstrip('.')),2), round(float(seed_max_cv_value.replace('_', '.').rstrip('.')),2), int(seed_mth_Gene_value)])
        

for max_cv in max_cvs:
    max_cv = round(max_cv, 2)
    for min_cv in min_cvs:
        min_cv = round(min_cv, 2)
        for min_r in min_regions:
            for eigen_max in eigen_maxs:
                for model_n in model_ns:
                    if min_cv >= max_cv:
                        continue

                    if [eigen_max, min_cv, max_cv, min_r] in param_sets:
                        print("{}, {}, {}, {} already included".format(eigen_max, min_cv, max_cv, min_r))
                        continue

                    print("{}, {}, {}, {} will be analyzed".format(eigen_max, min_cv, max_cv, min_r))

                    args = "{} {} {} {} {} {}".format(min_r, eigen_max, model_n,min_cv,max_cv, core)

                    cout = "julia "+j_dir + jl_f + " " + args
                    subprocess.run([cout], shell=True)