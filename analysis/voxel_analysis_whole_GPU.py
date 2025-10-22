import os
import gc
import math
import pickle
import subprocess

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from statsmodels.stats.multitest import multipletests

import cfospy
import costest

    
def rad2ph(rad):
    if math.isnan(rad):
        return np.nan
    else:
        return (round((2*np.pi+rad)*180/np.pi*24/360, 1),  round((rad)*180/np.pi*24/360, 1))[rad>=0]


class calc_vb_phase:
    def __init__(self, rdir, savedir, vb_pre_file, mask0, vx, r, vb_r, mo):
        self.region = "whole"
        self.vx = vx
        self.r = r
        self.savedir = savedir
        self.vb_r = vb_r
        self.mo = mo
        self.rdir = rdir

        region = self.region
        if "/" in region:
            region = region.replace("/", "_")
        print(region)

        # Total voxel number in mask
        self.total_b_num = len(np.where(np.ravel(mask0) == 1)[0])

        # Get bounding box of mask
        v_ind = np.where(mask0)
        xmin = np.min(v_ind[2])
        ymin = np.min(v_ind[1])
        zmin = np.min(v_ind[0])
        xmax = np.max(v_ind[2])
        ymax = np.max(v_ind[1])
        zmax = np.max(v_ind[0])

        # Pad bounding box
        xmin = int(xmin - (xmax - xmin) / r)
        ymin = int(ymin - (ymax - ymin) / r)
        zmin = int(zmin - (zmax - zmin) / r)
        xmax = int(xmax + (xmax - xmin) / r)
        ymax = int(ymax + (ymax - ymin) / r)
        zmax = int(zmax + (zmax - zmin) / r)

        # Clamp within atlas dimensions
        xmin = max(0, xmin)
        xmax = min(ca.x_num, xmax)
        ymin = max(0, ymin)
        ymax = min(ca.y_num, ymax)
        zmin = max(0, zmin)
        zmax = min(ca.z_num, zmax)

        # Store coordinates
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax

        vol = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
        print("voxel volume", vol)
        print("moving", self.mo)

        self.vb_pre_file = vb_pre_file

        self.root_dir_count = (
            f"{vb_pre_file}/{self.vx}um/{region}/vb{self.vb_r}_mo{self.mo}/"
        )

        if calc_type == "count_ratio":
            calc_dir = "whole_vb_new"
        elif calc_type == "count":
            calc_dir = "whole_vb_a_new"
        elif calc_type == "cell_intensity_ratio":
            calc_dir = "whole_vb_cir"
        elif calc_type == "cell_intensity":
            calc_dir = "whole_vb_ci"
        else:
            raise ValueError(f"Unknown calc_type: {calc_type}")

        self.calc_dir = calc_dir
        self.dir = f"/{calc_dir}/{self.vx}um/{region}/vb{self.vb_r}_mo{self.mo}/"

        self.dst = os.path.join(self.savedir, self.dir)
        os.makedirs(self.dst, exist_ok=True)
        print(self.dst)

    
    def get_vb_ID(self):
        # Build voxel index volume (x, y, z)
        img_vx_ID = np.arange(len(ca.voxel_ID_order_all)).reshape(ca.x_num, ca.y_num, ca.z_num)
        vb_ID = {}
        c = 0
    
        # Block counts along each axis
        self.x_b_num = int((self.xmax - self.xmin) / self.mo + 1)
        self.y_b_num = int((self.ymax - self.ymin) / self.mo + 1)
        self.z_b_num = int((self.zmax - self.zmin) / self.mo + 1)
        self.total_b_num = self.x_b_num * self.y_b_num * self.z_b_num
    
        pkl_path = os.path.join(self.dst, "vb_ID.pkl")
        if not os.path.exists(pkl_path):
            for x in range(self.x_b_num):
                for y in range(self.y_b_num):
                    for z in range(self.z_b_num):
                        if self.mo != 1:
                            # Compute block bounds with padding logic identical to original
                            if self.xmin + x * self.mo - self.mo + 1 < 0:
                                xmin2 = 0
                            else:
                                xmin2 = self.xmin + x * self.mo - self.mo + 1
    
                            if self.xmin + x * self.mo + self.mo - 1 > ca.x_num:
                                xmax2 = ca.x_num
                            else:
                                xmax2 = self.xmin + x * self.mo + self.mo - 1
    
                            if self.ymin + y * self.mo - self.mo + 1 < 0:
                                ymin2 = 0
                            else:
                                ymin2 = self.ymin + y * self.mo - self.mo + 1
    
                            if self.ymin + y * self.mo + self.mo - 1 > ca.y_num:
                                ymax2 = ca.y_num
                            else:
                                ymax2 = self.ymin + y * self.mo + self.mo - 1
    
                            if self.zmin + z * self.mo - self.mo + 1 < 0:
                                zmin2 = 0
                            else:
                                zmin2 = self.zmin + z * self.mo - self.mo + 1
    
                            if self.zmin + z * self.mo + self.mo - 1 > ca.z_num:
                                zmax2 = ca.z_num
                            else:
                                zmax2 = self.zmin + z * self.mo + self.mo - 1
    
                            vb_ID[c] = np.ravel(img_vx_ID[xmin2:xmax2, ymin2:ymax2, zmin2:zmax2])
                        else:
                            # Single-voxel step when mo == 1
                            if (
                                self.xmin + x * self.mo >= ca.x_num
                                or self.ymin + y * self.mo >= ca.y_num
                                or self.zmin + z * self.mo >= ca.z_num
                            ):
                                c += 1
                                continue
    
                            vb_ID[c] = img_vx_ID[
                                self.xmin + x * self.mo,
                                self.ymin + y * self.mo,
                                self.zmin + z * self.mo,
                            ]
    
                        c += 1
    
            with open(pkl_path, "wb") as tf:
                pickle.dump(vb_ID, tf)

        
    def make_cos_vb(self, sample_names):
        # Skip if output already exists
        csv_path = os.path.join(self.dst, "cos_1st2nd.csv")
        if not os.path.exists(csv_path):
            CT_li = np.arange(0, 48, 4)
            CT_li2 = np.arange(0, 96, 4)
            sample_ids = np.arange(1, 7, 1)
    
            vb_exp = np.zeros((self.total_b_num, len(CT_li) * len(sample_ids)), dtype="float32")
    
            # 1st series
            exp = "1st"
            exp_dir = os.path.join(self.savedir, exp, self.dir.lstrip("/"))
            os.makedirs(exp_dir, exist_ok=True)
    
            vb_ct1 = os.path.join(exp_dir, f"vb_CT_{exp}")
            if not os.path.exists(vb_ct1):
                for m, CT in enumerate(CT_li):
                    for n, sample_id in enumerate(sample_ids):
                        sample = f"CT{CT}_{str(sample_id).zfill(2)}"
                        print(sample)
                        cf = os.path.join(self.savedir, exp, self.root_dir_count, f"{sample}_vb_CT_{exp}.bin")
                        rectype = np.dtype(np.int32)
                        vb_bin = np.fromfile(cf, dtype=rectype)
                        print("vb_bin", vb_bin.shape)
                        vb_exp[:, m * len(sample_ids) + n] = vb_bin
    
                if calc_type == "count_ratio":
                    T_cells = np.load(os.path.join(self.savedir, exp, "total_cell_nums.npy"))
                    vb_exp = vb_exp / T_cells.reshape(-1, len(CT_li) * len(sample_ids))
                elif calc_type == "cell_intensity_ratio":
                    T_cells = np.load(os.path.join(self.savedir, exp, "total_cell_intenses.npy"))
                    vb_exp = vb_exp / T_cells.reshape(-1, len(CT_li) * len(sample_ids))
                elif calc_type == "count":
                    vb_exp = vb_exp
                elif calc_type == "cell_intensity":
                    vb_exp = vb_exp
    
                np.save(vb_ct1, vb_exp)
    
            # 2nd series
            exp = "2nd"
            exp_dir = os.path.join(self.savedir, exp, self.dir.lstrip("/"))
            os.makedirs(exp_dir, exist_ok=True)
    
            vb_ct2 = os.path.join(exp_dir, f"vb_CT_{exp}")
            if not os.path.exists(vb_ct2):
                for m, CT in enumerate(CT_li):
                    for n, sample_id in enumerate(sample_ids):
                        sample = f"CT{CT}_{str(sample_id).zfill(2)}"
                        print(sample)
                        cf = os.path.join(self.savedir, exp, self.root_dir_count, f"{sample}_vb_CT_{exp}.bin")
                        rectype = np.dtype(np.int32)
                        vb_bin = np.fromfile(cf, dtype=rectype)
                        print("vb_bin", vb_bin.shape)
                        vb_exp[:, m * len(sample_ids) + n] = vb_bin
    
                if calc_type == "count_ratio":
                    T_cells = np.load(os.path.join(self.savedir, exp, "total_cell_nums.npy"))
                    vb_exp = vb_exp / T_cells.reshape(-1, len(CT_li) * len(sample_ids))
                elif calc_type == "cell_intensity_ratio":
                    T_cells = np.load(os.path.join(self.savedir, exp, "total_cell_intenses.npy"))
                    vb_exp = vb_exp / T_cells.reshape(-1, len(CT_li) * len(sample_ids))
                elif calc_type == "count":
                    vb_exp = vb_exp
                elif calc_type == "cell_intensity":
                    vb_exp = vb_exp
    
                np.save(vb_ct2, vb_exp)
    
            # Concatenate 1st & 2nd, compute cosinor
            cols = []
            for m, CT in enumerate(CT_li2):
                for n, sample_id in enumerate(sample_ids):
                    sample = f"CT{CT}_{str(sample_id).zfill(2)}"
                    cols.append(sample)
    
            path1 = os.path.join(self.savedir, "1st", self.dir.lstrip("/"), "vb_CT_1st.npy")
            CT_np1 = np.load(path1)
    
            path2 = os.path.join(self.savedir, "2nd", self.dir.lstrip("/"), "vb_CT_2nd.npy")
            CT_np2 = np.load(path2)
    
            combined = np.hstack([CT_np1, CT_np2])
            del CT_np1, CT_np2
            gc.collect()
    
            df = pd.DataFrame(combined)
            del combined
            gc.collect()
            
            df.columns = cols
            df.insert(0, "id", np.arange(0, self.total_b_num))
    
            # Compute mean and SEM per time point
            avg_list = []
            se_list = []
            for i in range(df.shape[0]):
                vals = df.iloc[i, 1:(1 + 6 * 12 * 2)].to_numpy(dtype=int)  # 6 reps × 12 CT × 2 series = 144
                tbl = vals.reshape(24, 6).astype("uint32")
                avg = tbl.mean(axis=1).astype("float32")
                se = (tbl.std(axis=1, ddof=1) / np.sqrt(tbl.shape[1])).astype("float32")
                avg_list.append(avg)
                se_list.append(se)
        
            # Batch cosinor with SEM
            data_matrix = np.array(avg_list).astype("float32")
            se_matrix = np.array(se_list).astype("float32")
            del avg_list, se_list
            gc.collect()
    
            results = batch_costest(data_matrix[:, :], 6, se_matrix, alpha).astype("float32")
            del data_matrix, se_matrix
            gc.collect()
    
            id_list = df["id"].tolist()
            del df
            gc.collect()
    
            mc, mc_ph, p_org, p_sem_adj = results[:, 0], results[:, 1], results[:, 2], results[:, 3]
            nan_id = np.isnan(p_sem_adj).astype("uint32")
            nonan_id = (np.where(nan_id == False)[0]).astype("uint32")
            del nan_id, p_org
            gc.collect()
    
            phase_li = list(map(rad2ph, mc_ph))
            per_li = np.ones(len(phase_li), dtype="uint8") * 24
    
            cos_v_df = pd.DataFrame({
                "id": id_list,
                "ADJ.P": p_sem_adj,
                "PER": per_li,
                "Ph": mc_ph,
                "LAG": phase_li,
                "max_corr": mc,
            })
    
            del per_li, phase_li, mc, mc_ph
            gc.collect()
    
            q_bh_np = np.ones(len(p_sem_adj), dtype="float32")
            _, q_bh_li, _, _ = multipletests(p_sem_adj[nonan_id], method="fdr_bh")
            del p_sem_adj
            gc.collect()
    
            q_bh_np[nonan_id] = q_bh_li
            del q_bh_li, nonan_id
            gc.collect()
    
            cos_v_df.insert(3, "BH.Q", q_bh_np)
            del q_bh_np
            gc.collect()
    
            # Save result
            cos_v_df.to_csv(csv_path, index=False)

    
    def make_vb_image(self, atlas_mask, op="fdr", op_alpha=True, n=3, b=0.01):
        # Nonlinear alpha mapping parameter
        a = (1 - b) / ((-1) ** n)
    
        tiff_path = os.path.join(self.dst, f"vb_{op}_img.tif")
        if not os.path.exists(tiff_path):
            csv_path = os.path.join(self.savedir, self.dir.lstrip("/"), "cos_1st2nd.csv")
            cos_v_df = pd.read_csv(csv_path, index_col=0)
    
            img_vx = np.zeros((len(ca.voxel_ID_order_all), 4), dtype="uint8")
    
            if op == "fdr":
                fdr_li = cos_v_df["BH.Q"] < 0.1
                if op_alpha is True:
                    fdr_vs = np.array(cos_v_df["BH.Q"])
                    alpha_li = -np.log10(fdr_vs)
                    a_max = np.max(alpha_li)
                    alpha_li = alpha_li / a_max
                    alpha_li = (-a * (alpha_li - 1) ** n + 1) * 255
                    print("alpha_max", np.max(alpha_li))
                    print("alpha_min", np.min(alpha_li))
                else:
                    alpha_li = np.ones(len(cos_v_df)) * 255
            else:
                alpha_li = np.ones(len(cos_v_df)) * 255
    
            ph_li = (cos_v_df["LAG"] / 24)
            ph_li = 1 - ph_li
            ph_li = [(hue + 1 / 3 - 1) if (hue + 1 / 3) > 1 else (hue + 1 / 3) for hue in ph_li]
    
            brain_ind = np.where(np.ravel(np.swapaxes(atlas_mask, 0, 2)) == 1)[0]
    
            for i in range(len(cos_v_df)):
                vx_ind = brain_ind[i]
                if op == "fdr":
                    if fdr_li[i] == 1:
                        img_vx[vx_ind] = np.append(hsv_to_rgb([ph_li[i], 1, 1]) * 255, alpha_li[i])
                else:
                    img_vx[vx_ind] = np.append(hsv_to_rgb([ph_li[i], 1, 1]) * 255, alpha_li[i])
    
            img_vx = np.swapaxes(img_vx.reshape(ca.x_num, ca.y_num, ca.z_num, 4), 0, 2)
            print(img_vx.shape)
    
            os.makedirs(self.dst, exist_ok=True)
            tifffile.imwrite(tiff_path, img_vx)
            print("save: ", tiff_path)
        else:
            img_vx = tifffile.imread(os.path.join(self.dst, f"vb_{op}_img.tif"))
    
        return img_vx

    
    def make_nuc_atlas_image(self, ants_dir_name):
        # Generate normalized nuclear intensity atlas image
        after_ants_file = os.path.join(self.savedir, f"mean_nuc_img_{self.vx}um.tif")
    
        with tifffile.TiffFile(after_ants_file) as tif:
            img = tif.asarray()
    
        img_mask = img > 200
        mask_ind = np.where(img_mask == 1)
    
        # Normalize intensity to 8-bit depth
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
        img = img.astype(np.uint8)
        img[mask_ind] += 50
    
        return img
    
            
    def overlay_images_whole(self, atlas_mask, ants_dir_name, size, s,
                            op1="fdr", op2="nuc_atlas", op_alpha=True, n=3, b=0.01):
    
        # Prepare base image (phase/FDR) or atlas image depending on op2
        img_vx = self.make_vb_image(atlas_mask, "fdr", op_alpha, n, b)
    
        # Prepare overlay image according to op2
        r_img = self.make_nuc_atlas_image(ants_dir_name)
    
        sl_num = size * size + 2
        os.makedirs(self.dst, exist_ok=True)
    
        # Horizontal
        fig = plt.figure(figsize=(20, 20))
        slice_vs = np.linspace(self.zmin, self.zmax, sl_num)
        for i, sl in enumerate(slice_vs):
            if i == 0 or i == len(slice_vs) - 1:
                continue
            ax = fig.add_subplot(size, size, i)
            ax.imshow(r_img[int(sl), :, :])
            ax.imshow(img_vx[int(sl), :, :])
            ax.set_xlim(self.xmin + s, self.xmax - s)
            ax.set_ylim(self.ymax + s, self.ymin - s)
            ax.axis("off")
        plt.tight_layout()
        fname = (f"vb_{op1}_{op2}_slice_hor.png" if op_alpha is False
                 else f"vb_{op1}_{op2}_alphan{n}_b{b}_slice_hor.png")
        fig.savefig(os.path.join(self.dst, fname))

        # Coronal
        fig = plt.figure(figsize=(20, 20))
        slice_vs = np.linspace(self.ymin, self.ymax, sl_num)
        for i, sl in enumerate(slice_vs):
            if i == 0 or i == len(slice_vs) - 1:
                continue
            ax = fig.add_subplot(size, size, i)
            ax.imshow(r_img[:, int(sl), :])
            ax.imshow(img_vx[:, int(sl), :])
            ax.axis("off")
            ax.set_ylim(self.zmax + s, self.zmin - s)
            ax.set_xlim(self.xmin + s, self.xmax - s)
        plt.tight_layout()
        fname = (f"vb_{op1}_{op2}_slice_cor.png" if op_alpha is False
                 else f"vb_{op1}_{op2}_alphan{n}_b{b}_slice_cor.png")
        fig.savefig(os.path.join(self.dst, fname))

        # Sagittal
        fig = plt.figure(figsize=(20, 20))
        slice_vs = np.linspace(self.xmin, self.xmax, sl_num)
        for i, sl in enumerate(slice_vs):
            if i == 0 or i == len(slice_vs) - 1:
                continue
            ax = fig.add_subplot(size, size, i)
            ax.imshow(r_img[:, :, int(sl)])
            ax.imshow(img_vx[:, :, int(sl)])
            ax.axis("off")
            ax.set_ylim(self.zmax + s, self.zmin - s)
            ax.set_xlim(self.ymin + s, self.ymax - s)
        plt.tight_layout()
        fname = (f"vb_{op1}_{op2}_slice_sag.png" if op_alpha is False
                 else f"vb_{op1}_{op2}_alphan{n}_b{b}_slice_sag.png")
        fig.savefig(os.path.join(self.dst, fname))


if __name__ == '__main__':
    # Path settings
    src = "/path/to/source_dir"
    dst = "/path/to/output_dir"
    
    rdir = os.path.join(src, "CUBIC_R_atlas_ver5")
    savedir = os.path.join(dst, "cfos_app")
    cfos_dir = os.path.join(src, "circadian_1st", "circadian_1st_Reconst")

    # Analysis parameters
    vx = 20            # voxel size [µm]
    r = 100            # relative margin divisor
    vb_r = 8           # voxel block factor (convolution kernel = vx * vb_r)
    mo = 1             # moving offset
    s = 0              # slice offset
    size = 5           # number of slices per row/column
    vb_pre_file = "whole_vb_a_new"
    ants_dir_name = "ANTsR50"
    count_file = "cell_table_combine_I_ai_fpr0.5.npy"
    ncore = 20
    blockdim_x = 1
    calc_type = "count"
    
    # Collect unique sample names
    CT_li = np.arange(0, 48, 4)  # circadian time points (CT0–44, every 4 h)
    sample_ids = np.arange(1, 7, 1)
    
    reconsts = os.listdir(cfos_dir)
    sample_names = []
    
    for CT in CT_li:
        for sample_id in sample_ids:
            sample = f"CT{CT}_{str(sample_id).zfill(2)}"
            for reconst in reconsts:
                if sample in reconst:
                    sample_names.append(sample)
    
    print(f"{len(sample_names)} samples")    
    
    # Load atlas data
    ca = cfospy.analysis.read_atlas_data(rdir, vx)
    atlas_mask = ca.get_atlas_mask()
    print(f"{len(ca.ID_all)} regions")

    # Initialize voxel-block object
    vb = calc_vb_phase(rdir, savedir, vb_pre_file, atlas_mask, vx, r, vb_r, mo)
    xmin = vb.xmin
    xmax = vb.xmax
    ymin = vb.ymin
    ymax = vb.ymax
    zmin = vb.zmin
    zmax = vb.zmax
    
    # Build voxel-coordinates path
    vx_cords_f = os.path.join(rdir, f"{vx}um", "voxel_cords_brain.npy")
    
    # Check precomputed vb binary (2nd/...) and run external job if missing
    target_bin = os.path.join(vb.savedir, "2nd", vb.root_dir_count, "CT44_06_vb_CT_1st.bin")
    if not os.path.exists(target_bin):
        args = f"{vx} {vb_r} {mo} {vx_cords_f} {vb.savedir} {vb_pre_file} {ants_dir_name} {count_file} {ncore} {blockdim_x}"
        outc = ("./vx_c_w " if (calc_type == "count" or calc_type == "count_ratio") else "./vx_ci_w ") + args
        subprocess.run([outc], shell=True)
    
    # Compute cosinor table and overlay images
    vb.make_cos_vb(sample_names)
    vb.overlay_images_whole(atlas_mask, ants_dir_name, size, s)