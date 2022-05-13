import numpy as np

class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio): # (8,14,14), 0.9
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame =  self.height * self.width # 196
        self.total_patches = self.frames * self.num_patches_per_frame # 1568
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame) # int(196*0.9)=176
        self.total_masks = self.frames * self.num_masks_per_frame # 8*176=1408

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame), # 20
            np.ones(self.num_masks_per_frame), # 176
        ]) # np.array([0]*20+[1]*176)
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames,1)).flatten() # [0,1,...] # length=8*196=1568
        return mask 
