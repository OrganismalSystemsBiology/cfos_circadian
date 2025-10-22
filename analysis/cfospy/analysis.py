import os
import numpy as np
import pandas as pd
import tifffile

# Get absolute path of this file
current_file = os.path.abspath(__file__)

# Get directory containing this file
current_dir = os.path.dirname(current_file)

# Get parent directory
parent_dir = os.path.dirname(current_dir)
print(parent_dir)

class read_atlas_data:
    def __init__(self, rdir, vx):
        self.rdir = rdir
        self.vx = vx
        self.df_allen = pd.read_csv(os.path.join(parent_dir, "Allen_ID_all.csv"))
        # Remove root and grey
        self.ID_all = self.df_allen["ID"].iloc[0:].tolist()[2:]

        if os.path.exists(os.path.join(rdir, f"{vx}um", "voxel_ID_order.npy")):
            self.voxel_ID_order_all = np.load(
                os.path.join(rdir, f"{vx}um", "voxel_ID_order_all.npy")
            )
        else:
            print("voxel_ID_order_file not found")
            print(os.path.join(rdir, f"{vx}um", "voxel_ID_order_all.npy"))
        
        self.voxel_nums = len(self.voxel_ID_order_all)
        print("voxel_num:", len(self.voxel_ID_order_all))
    
    
    def smallID_q(self, rID):
        """Return True if rID is a 'small' region (i.e., has no child regions)."""
        trees = self.df_allen["tree"]
        small_flag0 = True
        for _, tree in enumerate(trees):
            parts = tree.split("/")
            if str(rID) in parts and parts[-2] != str(rID):
                small_flag0 = False
                break
                
        return small_flag0
    
    
    def get_child_IDs(self, rID):
        """
        Return child region IDs and acronyms for rID.
        If rID is already a 'small' region, return itself.
        """
        region = self.df_allen[self.df_allen["ID"] == rID]["acronym"].iloc[0]

        trees = self.df_allen["tree"]
        child_IDs = []
        child_regions = []
        small_flag0 = True

        for _, tree in enumerate(trees):
            parts = tree.split("/")
            if str(rID) in parts and parts[-2] != str(rID):
                small_flag0 = False
                break

        if small_flag0 is False:
            for i, tree in enumerate(trees):
                parts = tree.split("/")
                if str(rID) in parts:
                    ch_ID = self.df_allen.iloc[i]["ID"]
                    ch_region = self.df_allen.iloc[i]["acronym"]
                    small_flag = True
                    for _, t2 in enumerate(trees):
                        parts2 = t2.split("/")
                        if str(ch_ID) in parts2 and parts2[-2] != str(ch_ID):
                            small_flag = False
                            break
                    if small_flag is True:
                        child_IDs.append(ch_ID)
                        child_regions.append(ch_region)
        else:
            # rID is already a small region
            child_IDs.append(rID)
            child_regions.append(region)
        
        return child_IDs, child_regions
    
    
    def get_child_IDs2(self, rID):
        """Return small child region IDs/acronyms and also non-small descendants (excluding the first element)."""
        region = self.df_allen[self.df_allen["ID"] == rID]["acronym"].iloc[0]
    
        trees = self.df_allen["tree"]
        child_IDs = []
        child_regions = []
        small_flag0 = True
    
        for _, tree in enumerate(trees):
            parts = tree.split("/")
            if str(rID) in parts and parts[-2] != str(rID):
                small_flag0 = False
                break
    
        if small_flag0 is False:
            not_small = []
            not_small_IDs = []
            for i, tree in enumerate(trees):
                parts = tree.split("/")
                if str(rID) in parts:
                    ch_ID = self.df_allen.iloc[i]["ID"]
                    ch_region = self.df_allen.iloc[i]["acronym"]
                    small_flag = True
                    for _, t2 in enumerate(trees):
                        parts2 = t2.split("/")
                        if str(ch_ID) in parts2 and parts2[-2] != str(ch_ID):
                            if ch_region not in not_small:
                                not_small.append(ch_region)
                                not_small_IDs.append(ch_ID)
                            small_flag = False
                            break
                    if small_flag is True:
                        child_IDs.append(ch_ID)
                        child_regions.append(ch_region)
        else:
            # rID is already a small region
            child_IDs.append(rID)
            child_regions.append(region)
        
        return child_IDs, child_regions, not_small_IDs[1:], not_small[1:]
    
    
    def region_periodic(self, CT_df, rID, th):
        """Return periodic regions under rID and the percentage within all descendants (incl. rID)."""
        region = CT_df[CT_df["id"] == rID]["acronym"].iloc[0]
        print(region)
    
        per_ID = CT_df[CT_df["BH.Q"] < th]["id"].tolist()
        child_IDs, child_regions, middle_IDs, middle_regions = self.get_child_IDs2(rID)
    
        print("num of small regions ", len(child_regions))
        print("num of middle regions", len(middle_regions))
    
        region_all_ID = [rID] + middle_IDs + child_IDs
        print("num of total regions", len(region_all_ID))
    
        # significant periodic regions
        region_per = [i for i in per_ID if i in region_all_ID]
        print("number of periodic regions in {}".format(region), len(region_per))
    
        ratio_per = len(region_per) / len(region_all_ID) * 100
        print("% of periodic regions", ratio_per)
        
        return region_per, ratio_per
    
    
    def get_uni_rIDs(self):
        """Return unique region IDs to analyze and the list of removed (insufficient) IDs."""
        uni_IDs = []
        rev_IDs = []
    
        for _, rID in enumerate(self.ID_all):
            if self.smallID_q(rID):
                index = (np.where(self.voxel_ID_order_all == rID)[0])
                if len(index) != 0:
                    uni_IDs.append(rID)
            else:
                if len(np.where(self.voxel_ID_order_all == rID)[0]) != 0:
                    uni_IDs.append(rID)
                else:
                    child_IDs, child_regions, middle_IDs, middle_regions = self.get_child_IDs2(rID)
                    rID_flag = 0
                    m_ind = 0
                    for m_ID in middle_IDs + child_IDs:
                        if len(np.where(self.voxel_ID_order_all == m_ID)[0]) != 0:
                            m_ind += 1
                            if m_ind == 2:
                                uni_IDs.append(rID)
                                rID_flag = 1
                                break
                    if rID_flag == 0:
                        rev_IDs.append(rID)
    
        uni_IDs.append(1051)  # tspd
    
        region_num = len(uni_IDs)
        print("region_num:", region_num)
        print("remove IDs:", rev_IDs)
        
        return uni_IDs, rev_IDs
    
    
    def get_vx_ind(self, rID):
        """Return voxel indices belonging to rID (including eligible descendants)."""
        if self.smallID_q(rID):
            index = (np.where(self.voxel_ID_order_all == rID)[0]).tolist()
        else:
            index = []
            index += (np.where(self.voxel_ID_order_all == rID)[0]).tolist()
            child_IDs, child_regions, middle_IDs, middle_regions = self.get_child_IDs2(rID)
            for m_ID in middle_IDs + child_IDs:
                index += (np.where(self.voxel_ID_order_all == m_ID)[0]).tolist()
                
        return index
    
    
    def get_center(self, rID):
        """Return geometric center (z, y, x) of rID mask."""
        center = np.zeros(3, dtype="float32")
        index = self.get_vx_ind(rID)
        vx_ind = np.zeros(len(self.voxel_ID_order_all), dtype="uint8")
        vx_ind[index] = 1
        img_mask = np.swapaxes(vx_ind.reshape(self.x_num, self.y_num, self.z_num), 0, 2)
        r_cords = np.where(img_mask == 1)
        center[2] = np.mean(r_cords[2])
        center[1] = np.mean(r_cords[1])
        center[0] = np.mean(r_cords[0])
        
        return center
    
    
    def get_r_center(self, rID):
        """Return right-half geometric center (z, y, x) of rID mask."""
        center = np.zeros(3, dtype="float32")
        index = self.get_vx_ind(rID)
        vx_ind = np.zeros(len(self.voxel_ID_order_all), dtype="uint8")
        vx_ind[index] = 1
        img_mask = np.swapaxes(vx_ind.reshape(self.x_num, self.y_num, self.z_num), 0, 2)
        r_cords = np.where(img_mask == 1)
        x_half = [i for i in r_cords[2] if i > self.x_num / 2]
        center[2] = np.mean(x_half)
        center[1] = np.mean(r_cords[1])
        center[0] = np.mean(r_cords[0])
        
        return center
    
    
    def get_lr_center(self, rID):
        """Return (left_center, right_center) as (z, y, x)."""
        center_r = np.zeros(3, dtype="float32")
        center_l = np.zeros(3, dtype="float32")
        index = self.get_vx_ind(rID)
        vx_ind = np.zeros(len(self.voxel_ID_order_all), dtype="uint8")
        vx_ind[index] = 1
        img_mask = np.swapaxes(vx_ind.reshape(self.x_num, self.y_num, self.z_num), 0, 2)
        r_cords = np.where(img_mask == 1)
    
        x_half = [i for i in r_cords[2] if i > self.x_num / 2]
        center_r[2] = np.mean(x_half)
        center_r[1] = np.mean(r_cords[1])
        center_r[0] = np.mean(r_cords[0])
    
        x_half = [i for i in r_cords[2] if i <= self.x_num / 2]
        center_l[2] = np.mean(x_half)
        center_l[1] = np.mean(r_cords[1])
        center_l[0] = np.mean(r_cords[0])
        
        return center_l, center_r
    
    
    def get_sum_temp(self, uni_IDs):
        """Return a summary DataFrame filtered by uni_IDs and store volume order."""
        ex_file = "ex_summary.csv"
        df_ex = pd.read_csv(os.path.join(self.rdir, ex_file))
        self.volume_order = np.array([df_ex[df_ex["id"] == rID]["volume"] for rID in uni_IDs])
        df_sum = df_ex[df_ex["id"].isin(uni_IDs)].iloc[:, 0:6]
        print(df_sum)
        
        return df_sum
    
    
    def get_atlas_img(self):
        """Load atlas image and set x/y/z dimensions."""
        atlas_path = os.path.join(self.rdir, f"iso_{self.vx}um_R.tif")
        atlas_img = tifffile.imread(atlas_path)
        self.x_num = atlas_img.shape[2]
        self.y_num = atlas_img.shape[1]
        self.z_num = atlas_img.shape[0]
        
        return atlas_img

    
    def get_atlas_mask(self):
        """Load atlas mask and set x/y/z dimensions."""
        id_file = f"CUBIC_R_space_{self.vx}um_annotation_ver5.tif"
        img_ID = tifffile.imread(os.path.join(self.rdir, id_file))
        atlas_mask = img_ID > 0
        self.x_num = atlas_mask.shape[2]
        self.y_num = atlas_mask.shape[1]
        self.z_num = atlas_mask.shape[0]
        
        return atlas_mask