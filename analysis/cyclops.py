import os
import re
import subprocess

dst = "/path/to/output_dir"

# Julia template filename
jl_f = "CYCLOPS_2_0_Template.jl"

j_dir = os.path.join(dst, "cfos_app", "timetable", "CYCLOPS-2.0-main")
results_pre_dir = os.path.join(j_dir, "results")
jl_path = os.path.join(j_dir, jl_f)
files = os.listdir(results_pre_dir)

# Parameter grids
min_regions = [20]
eigen_maxs = [9]
model_ns = [80]
min_cvs = [0.15]
max_cvs = [0.8]
core = 20

# Default tokens
eigen_max_d = "99"
mincv_d = "0_85_"
maxcv_d = "0_1_"

param_sets = []
for text in files:
    # Extract number after 'seed_min_CV_'
    seed_min_cv_match = re.search(r'seed_min_CV_([\d_]+)', text)
    seed_min_cv_value = seed_min_cv_match.group(1) if seed_min_cv_match else mincv_d

    # Extract number after 'eigen_max_'
    eigen_max_match = re.search(r'eigen_max_(\d+)', text)
    eigen_max_value = eigen_max_match.group(1) if eigen_max_match else eigen_max_d

    # Extract number after 'seed_max_CV_'
    seed_max_cv_match = re.search(r'seed_max_CV_([\d_]+)', text)
    seed_max_cv_value = seed_max_cv_match.group(1) if seed_max_cv_match else maxcv_d

    # Extract number after 'seed_mth_Gene_'
    seed_mth_Gene_match = re.search(r'seed_mth_Gene_([\d_]+)', text)
    seed_mth_Gene_value = seed_mth_Gene_match.group(1) if seed_mth_Gene_match else mincv_d

    param_sets.append([
        int(eigen_max_value),
        round(float(seed_min_cv_value.replace('_', '.').rstrip('.')), 2),
        round(float(seed_max_cv_value.replace('_', '.').rstrip('.')), 2),
        int(seed_mth_Gene_value)
    ])

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

                    args = "{} {} {} {} {} {}".format(min_r, eigen_max, model_n, min_cv, max_cv, core)
                    cmd = "julia {} {}".format(jl_path, args)
                    subprocess.run([cmd], shell=True)