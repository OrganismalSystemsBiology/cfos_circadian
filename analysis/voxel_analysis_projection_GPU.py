import os
import gc
import math
import pickle
import subprocess
import traceback

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
        

    def make_vb_image_edge(self, rID, img_vx, mask, lr, op_pre, op1, op2):
        # Build region edge RGBA image
        r_edge = np.zeros((ca.z_num, ca.y_num, ca.x_num, 4), dtype="uint8")
        
        edge_mask = tifffile.imread(os.path.join(self.rdir, f"{self.vx}um", "edge_mask.tif"))
        edge_vx_ind = np.where(edge_mask == 1)
        r_edge[edge_vx_ind] = (255, 255, 255, 100)  # other regions border

        edge_list = make_edge(rID, mask)
        for z, y, x, _ in edge_list:
            r_edge[z, y, x] = (255, 255, 255, 255)  # border of ROI
        r_edge[edge_mask == 0] = (0, 0, 0, 0)

        center_l, center_r = ca.get_lr_center(rID)

        # Load base region image
        if lr == "left":
            center = center_l
            r_img_path = os.path.join(self.savedir, "1st", f"region_crop_atlasR{self.vx}_{op_pre.replace('_lr','')}_left_maxp_mean_{self.angle}", self.region, "CT0.tif")
        else:
            center = center_r
            r_img_path = os.path.join(self.savedir, "1st", f"region_crop_atlasR{self.vx}_{op_pre.replace('_lr','')}_right_maxp_mean_{self.angle}", self.region, "CT0.tif")
    
        r_img = tifffile.imread(r_img_path)
    
        # Slice by view angle
        if self.angle == "hor":
            img_vx_c = img_vx[self.zmin, int(self.ymin):int(self.ymax), int(self.xmin):int(self.xmax), :]
            r_edge = r_edge[int(center[0]), int(self.ymin):int(self.ymax), int(self.xmin):int(self.xmax), :]
        elif self.angle == "cor":
            img_vx_c = img_vx[int(self.zmin):int(self.zmax), self.ymin, int(self.xmin):int(self.xmax), :]
            r_edge = r_edge[int(self.zmin):int(self.zmax), int(center[1]), int(self.xmin):int(self.xmax), :]
        elif self.angle == "sag":
            img_vx_c = img_vx[int(self.zmin):int(self.zmax), int(self.ymin):int(self.ymax), self.xmin, :]
            r_edge = r_edge[int(self.zmin):int(self.zmax), int(self.ymin):int(self.ymax), int(center[2]), :]
    
        print("r_img", r_img.shape)
        print("r_edge", r_edge.shape)
        print("img_vx_c", img_vx_c.shape)
    
        plt.figure(figsize=(5, 5))
        plt.imshow(r_img)
        plt.imshow(img_vx_c)
        plt.imshow(r_edge)
        plt.axis("off")
    
        out_dir = os.path.join(self.savedir, self.dir)
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f"vb_{op1}_{op2}_boder_slice_{self.angle}2_new.png"))
        plt.savefig(os.path.join(out_dir, f"vb_{op1}_{op2}_boder_slice_{self.angle}2_new.SVG"))
        print("save", os.path.join(out_dir, f"vb_{op1}_{op2}_boder_slice_{self.angle}_new.png"))
                                
    
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
        
        
    def overlay_images(self, s, op1="fdr", op2="border", op_alpha=True, n=3, b=0.01):
        # Create overlay inputs
        img_vx = self.make_vb_image("fdr", op_alpha, n, b)
        r_img = self.make_nuc_atlas_image()
    
        # Plot overlays by viewing angle
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(1, 1, 1)
    
        if self.angle == "hor":
            ax.imshow(np.max(r_img[self.zmin:self.zmax, :, :], axis=0))
            ax.imshow(img_vx[self.zmin + 5, :, :])
            ax.set_xlim(self.xmin + s, self.xmax - s)
            ax.set_ylim(self.ymax - s, self.ymin + s)
        elif self.angle == "cor":
            ax.imshow(np.max(r_img[:, self.ymin:self.ymax, :], axis=1))
            ax.imshow(img_vx[:, self.ymin + 5, :])
            ax.set_ylim(self.zmax - s, self.zmin + s)
            ax.set_xlim(self.xmin + s, self.xmax - s)
        elif self.angle == "sag":
            ax.imshow(np.max(r_img[:, :, self.xmin:self.xmax], axis=2))
            ax.imshow(img_vx[:, :, self.xmin + 5])
            ax.set_ylim(self.zmax - s, self.zmin + s)
            ax.set_xlim(self.ymin + s, self.ymax - s)
    
        ax.axis("off")
        plt.tight_layout()
    
        out_dir = os.path.join(self.savedir, self.dir)
        os.makedirs(out_dir, exist_ok=True)
        
        if op_alpha == False:
            out_path = os.path.join(out_dir, f"vb_{op1}_{op2}_slice_{self.angle}.png")
        else:
            out_path = os.path.join(out_dir, f"vb_{op1}_{op2}_alphan{n}_b{b}_slice_{self.angle}.png")
            
        fig.savefig(out_path)
    
        return img_vx


if __name__ == '__main__':
    # Path settings
    src = "/path/to/source_dir"
    dst = "/path/to/output_dir"
    
    rdir = os.path.join(src, "CUBIC_R_atlas_ver5")
    savedir = os.path.join(dst, "cfos_app")
    res = "cos.cell_count_1st2nd_ai_fpr0.5.csv"
    
    # Analysis parameters
    vx = 20            # voxel size [µm]
    r = 100            # relative margin divisor
    vb_r = 8           # voxel block factor (convolution kernel = vx * vb_r)
    mo = 1             # moving offset
    s = 0              # slice offset
    
    gpu_num = 0
    args = sys.argv
    
    rIDs = [286]
    l_ID = None
    angles = ["sag"]
    ants_dir_name_point_file = "ANTsR50"
    intensity_file = "cell_table_combine_I_ai_fpr0.5.npy"
    ncore = 10
    blockdim_x = blockdim_y = blockdim_z = 8
    calc_type = "count"
    op_pre = "RegionPlusBorder200_lr"
    
    if calc_type == "count":
        calc_dir_l = "region_vb_pro_c_l" 
        calc_dir_r = "region_vb_pro_c_r" 
    elif calc_type == "count_ratio":
        calc_dir_l = "region_vb_pro_cr_l" 
        calc_dir_r = "region_vb_pro_cr_r" 
    elif calc_type == "cell_intensity_ratio":
        calc_dir_l = "region_vb_pro_cir_l" 
        calc_dir_r = "region_vb_pro_cir_r"
    elif calc_type == "cell_intensity":
        calc_dir_l = "region_vb_pro_ci_l" 
        calc_dir_r = "region_vb_pro_ci_r"
    
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
    
    print(len(sample_names))
    
    # Load atlas data
    ca = cfospy.analysis.read_atlas_data(rdir, vx)
    atlas_mask = ca.get_atlas_mask()
    uni_IDs,rev_IDs = ca.get_uni_rIDs()
    print(f"{len(ca.ID_all)} regions")
    print(ca.x_num)
    print(ca.y_num)
    print(ca.z_num)
    del atlas_mask
    gc.collect()

    # If a parent ID is provided, expand to children/middle IDs
    if l_ID is not None:
        ch_IDs, ch_rs, m_IDs, m_rs = ca.get_child_IDs2(l_ID)
        rIDs = ch_IDs + m_IDs + [l_ID]
    
    # Mid-sagittal plane and offsets
    xc = int(ca.x_num / 2)
    xcoff = 5.0 * 20 / vx
    zoff = 25 * 20 / vx
    yoff = 10 * 20 / vx
    xoff = -10 * 20 / vx

    for i, rID in enumerate(rIDs):
        region = ca.df_allen[ca.df_allen["ID"] == rID]["acronym"].iloc[0]
        print(region)
        print("rID", rID)
    
        if "/" in region:
            region = region.replace("/", "_")

        # Collect IDs for this region (include children for large regions)
        if not ca.smallID_q(rID):
            ID_li = [rID]
            child_IDs, child_regions, middle_IDs, middle_regions = ca.get_child_IDs2(rID)
            for m_ID in child_IDs + middle_IDs:
                ID_li.append(m_ID)
        else:
            ID_li = [rID]
    
        # Build region mask
        mask0 = np.isin(np.swapaxes((ca.voxel_ID_order_all).reshape(ca.x_num, ca.y_num, ca.z_num), 0, 2), ID_li)

        v_ind = np.where(mask0)
        v_ind_xl = v_ind[2][np.where(v_ind[2] < xc)[0]]
        print(region, "vx_ind left", len(v_ind_xl))
        v_ind_xr = v_ind[2][np.where(v_ind[2] >= xc)[0]]
        print(region, "vx_ind_right", len(v_ind_xr))
        
        if len(v_ind_xl) == 0:
            xmin_l_o = 0
            xmax_l_o = 0
        else:
            xmin_l_o = np.min(v_ind_xl)
            xmax_l_o = np.max(v_ind_xl)
    
        if len(v_ind_xr) == 0:
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
    
            if angle == "hor":
                if rID == 286:
                    zmin = int(8632.242 / vx)
                    zmax = int(9992.238 / vx - zoff)
                    xmin_l = int(7363.42 / vx)
                    xmax_l = int(xc + xcoff)
                    ymin = int(9596.6045 / vx)
                    xmin_r = int(xc + xcoff)
                    xmax_r = int(9863.196 / vx)
                    ymax = int(12207.079 / vx)
                else:
                    xmin_l = int(xmin_l_o - (xmax_l_o - xmin_l_o) / r)
                    xmin_r = int(xmin_r_o - (xmax_r_o - xmin_r_o) / r)
                    ymin = int(ymin_o - (ymax_o - ymin_o) / r)
                    zmin = zmin_o
                    xmax_l = int(xmax_l_o + (xmax_l_o - xmin_l_o) / r)
                    xmax_r = int(xmax_r_o + (xmax_r_o - xmin_r_o) / r)
                    ymax = int(ymax_o + (ymax_o - ymin_o) / r)
                    zmax = zmax_o
    
            elif angle == "cor":
                if rID == 286:
                    ymin = int(9596.6045 / vx + yoff + 15 * 20 / vx)
                    ymax = int(12207.079 / vx - yoff - 50 * 20 / vx)
                    xmin_l = int(7363.42 / vx)
                    xmax_l = int(xc + xcoff)
                    xmin_r = int(xc + xcoff)
                    xmax_r = int(9863.196 / vx)
                    zmin = int(8519.214 / vx)
                    zmax = int(10133.864 / vx)
                else:
                    xmin_l = int(xmin_l_o - (xmax_l_o - xmin_l_o) / r)
                    xmin_r = int(xmin_r_o - (xmax_r_o - xmin_r_o) / r)
                    ymin = ymin_o
                    zmin = int(zmin_o - (zmax_o - zmin_o) / r)
                    xmax_l = int(xmax_l_o + (xmax_l_o - xmin_l_o) / r)
                    xmax_r = int(xmax_r_o + (xmax_r_o - xmin_r_o) / r)
                    ymax = ymax_o
                    zmax = int(zmax_o + (zmax_o - zmin_o) / r)
    
            elif angle == "sag":
                if rID == 286:
                    xmin_l = int(7906.6484 / vx + xoff)
                    xmax_l = int(xc + xcoff)
                    xmin_r = int(xc + xcoff)
                    xmax_r = int(9027.3530 / vx - xoff)
                    ymin = int(9596.6045 / vx)
                    ymax = int(12207.079 / vx)
                    zmin = int(8519.214 / vx)
                    zmax = int(10133.864 / vx)
                else:
                    xmin_l = xmin_l_o
                    xmin_r = xmin_r_o
                    ymin = int(ymin_o - (ymax_o - ymin_o) / r)
                    zmin = int(zmin_o - (zmax_o - zmin_o) / r)
                    xmax_l = xmax_l_o
                    xmax_r = xmax_r_o
                    ymax = int(ymax_o + (ymax_o - ymin_o) / r)
                    zmax = int(zmax_o + (zmax_o - zmin_o) / r)

            # Make L/R widths consistent
            if xmax_l - xmin_l < xmax_r - xmin_r:
                xmin_r = xmax_r - xmax_l + xmin_l
            elif xmax_l - xmin_l > xmax_r - xmin_r:
                xmax_l = xmax_r - xmin_r + xmin_l
    
            # Clamp to bounds
            if xmin_l < 0:
                xmin_l = 0
            if xmin_l > xc and rID != 286:
                xmin_l = xc
            if xmax_r > ca.x_num:
                xmax_r = ca.x_num
            if xmin_r < xc and rID != 286:
                xmin_r = xc
            if ymin < 0:
                ymin = 0
            if ymax > ca.y_num:
                ymax = ca.y_num
            if zmin < 0:
                zmin = 0
            if zmax > ca.z_num:
                zmax = ca.z_num

            if rID == 286:  # SCH
                if xmax_l - xmin_l < xmax_r - xmin_r:
                    xmax_r = xmin_r + xmax_l - xmin_l
                elif xmax_l - xmin_l > xmax_r - xmin_r:
                    xmin_l = -xmax_r + xmin_r + xmax_l

            # Left
            lr = "left"
            try:
                if len(v_ind_xl) != 0:
                    vb_l = calc_vb_phase(
                        rdir, savedir, calc_dir_l, rID, region, vx, r, vb_r, angle,
                        xmin_l, xmax_l, ymin, ymax, zmin, zmax
                    )
    
                    if not vb_l.mox == 0 and not vb_l.moy == 0 and not vb_l.moz == 0:
                        vb_l.get_vb_ID()
    
                        if vb_l.x_b_num < blockdim_x:
                            blockdim_x = 1
                        if vb_l.y_b_num < blockdim_y:
                            blockdim_y = 1
                        if vb_l.z_b_num < blockdim_z:
                            blockdim_z = 1

                        bin_path_l = os.path.join(vb_l.savedir, "2nd", vb_l.root_dir_count, f"CT44_06_vb_CT_{'2nd'}.bin")
                        if not os.path.exists(bin_path_l):
                            args = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format(
                                vb_l.x_b_num, vb_l.y_b_num, vb_l.z_b_num, xmin_l, ymin, zmin, xmax_l, ymax, zmax,
                                region, vx, vb_r, vb_l.mox, vb_l.moy, vb_l.moz, r, angle,
                                vb_l.savedir, vb_l.calc_dir, ants_dir_name_point_file, intensity_file,
                                ncore, blockdim_x, blockdim_y, blockdim_z
                            )
                            if calc_type == "count" or calc_type == "count_ratio":
                                outc = "env CUDA_VISIBLE_DEVICES={} ./vx_pro_c2 ".format(gpu_num) + args
                            else:
                                outc = "env CUDA_VISIBLE_DEVICES={} ./vx_pro_ci ".format(gpu_num) + args
                            print(outc)
                            subprocess.run([outc], shell=True)
    
                        vb_l.make_cos_vb(sample_names)
                        img_vx = vb_l.overlay_images(s)
                        vb_l.make_vb_image_edge(rID, img_vx, mask0, lr, op_pre, "fdr", "nuc_atlas")
                    else:
                        print(region, " left, mo is 0")
                        
                    del vb_l
                    gc.collect()
                else:
                    print(region, " no voxel in left")
            except:
                traceback.print_exc()
                
            # Right
            lr = "right"
            if rID == 286:
                if angle == "hor" or angle == "cor":
                    xmin_r = xmin_r - 5
            try:
                if len(v_ind_xr) != 0:
                    vb_right = calc_vb_phase(rdir, savedir, calc_dir_r, rID, region, vx, r, vb_r, angle, xmin_r, xmax_r, ymin, ymax, zmin, zmax)

                    if not vb_right.mox == 0 and not vb_right.moy == 0 and not vb_right.moz == 0:
                        vb_right.get_vb_ID()
    
                        if vb_right.x_b_num < blockdim_x:
                            blockdim_x = 1
                        if vb_right.y_b_num < blockdim_y:
                            blockdim_y = 1
                        if vb_right.z_b_num < blockdim_z:
                            blockdim_z = 1
    
                        bin_path_r = os.path.join(vb_right.savedir, "2nd", vb_right.root_dir_count, f"CT44_06_vb_CT_{'2nd'}.bin")
                        
                        if not os.path.exists(bin_path_r):
                            args = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format(
                                vb_right.x_b_num, vb_right.y_b_num, vb_right.z_b_num, xmin_r, ymin, zmin, xmax_r, ymax, zmax,
                                region, vx, vb_r, vb_right.mox, vb_right.moy, vb_right.moz, r, angle,
                                vb_right.savedir, vb_right.calc_dir, ants_dir_name_point_file, intensity_file,
                                ncore, blockdim_x, blockdim_y, blockdim_z
                            )
                            if calc_type == "count" or calc_type == "count_ratio":
                                outc = "env CUDA_VISIBLE_DEVICES={} ./vx_pro_c2 ".format(gpu_num) + args
                            else:
                                outc = "env CUDA_VISIBLE_DEVICES={} ./vx_pro_ci ".format(gpu_num) + args
                            print(outc)
                            subprocess.run([outc], shell=True)
    
                        vb_right.make_cos_vb(sample_names)
                        img_vx = vb_right.overlay_images(s)
                        vb_right.make_vb_image_edge(rID, img_vx, mask0, lr, op_pre, "fdr", "nuc_atlas")
                    else:
                        print(region, " right, mo is 0")
    
                    del vb_right
                    gc.collect()
                else:
                    print(region, " no voxel in right")
            except:
                traceback.print_exc()