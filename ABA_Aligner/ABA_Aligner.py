from numpy import ndarray, array, deg2rad, fliplr, flipud, rot90
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from matplotlib.widgets import Slider, Button
from skimage import io, transform
from skimage.transform import AffineTransform, warp, rescale
from skimage.morphology import dilation, disk
from tkinter import filedialog
from SPSIG import SPSIG

class ImageLoader:
    def __init__(self, folder:str = ''):
        # Find image files
        # self.locate_files()

        # Load your images (update the file paths as needed)
        atlas_raw = io.imread('ABA-Mask.png', as_gray = True)          # brain atlas outlines image, 240.51 x 240.42 pt
        coords_raw = io.imread('examples/Beta/AllenBrainAtlasOverlay_Beta_CoordinatesXY.png')
        twophoton_raw = io.imread('examples/Beta/AllenBrainAtlasOverlay_Beta_Original.png') # 1x zoom: 75,2019 x 75,2019 pt; 1.3x zoom: 57,8476 x 57,8476 pt; 1.6x zoom: 47 x 47 pt
        
        self.widefield = rot90(io.imread('examples/Beta/pRF_fieldsign.jpg'), k = -1)    # widefield image 334 x 389,0883 pt
        self.atlas = self.resize_image_pt(self.widefield, array([389.0883, 334]),
                                          atlas_raw, array([240.42, 240.51]))
        self.coords = self.resize_image_pt(self.widefield, array([389.0883, 334]),
                                          coords_raw, array([240.42, 240.51]))
        twop_mag = array([57.8476, 57.8476]) #self.find_2p_magnification()
        
        self.twophoton = self.resize_image_pt(self.widefield, array([389.0883, 334]),
                                              twophoton_raw, twop_mag)
            
    def resize_image_pt(self, reference_image:ndarray, reference_pt_dim:ndarray[float, float],
                    target_image:ndarray, target_pt_dim:ndarray[float, float])->ndarray:
        # ratio the images should be in
        ratio = target_pt_dim / reference_pt_dim
        
        reference_pix = reference_image.shape[:2]
        target_pix_goal = reference_pix * ratio
        
        return transform.resize(target_image, output_shape=target_pix_goal, anti_aliasing=True)
    
    # TODO
    def find_2p_magnification()->ndarray[float, float]:
        pass
    
    # TODO
    def locate_files(folder:str) -> tuple[str, str, str, str]:
        pass



class AtlasWidefieldGUI(ImageLoader):
    def __init__(self):
        super().__init__()
        # breakpoint()
        # Initial transformation parameters
        init_angle = 0       # in degrees
        init_tx = self.widefield.shape[1] // 2 - self.atlas.shape[1] // 2
        init_ty = self.widefield.shape[0] // 2 - self.atlas.shape[0] // 2

        # Create the figure and axes
        self.fig = plt.figure(figsize = (10,8))

        # Display the widefield image
        ax_main = self.fig.add_axes([0.05, 0.2, 0.9, 0.75])  # left=5%, bottom=20%, width=90%, height=75%
        ax_main.imshow(self.widefield)
        ax_main.set_title("Adjust Atlas Outlines")
        # ax_main.axis('off')

        # Apply initial transformation to the atlas outlines and overlay it
        atlas_transformed = self.transform_atlas(self.atlas, init_angle, init_tx, init_ty, self.widefield.shape[:2])
        self.overlay = ax_main.imshow(self.atlas_rgba_mask(atlas_transformed))

        # Define slider axes positions
        slider_height = 0.03
        slider_width = 0.7
        left_margin = 0.15
        bottom_start = 0.05
        ax_angle = self.fig.add_axes([left_margin, bottom_start + 2*slider_height, slider_width, slider_height])
        ax_tx    = self.fig.add_axes([left_margin, bottom_start + 1*slider_height, slider_width, slider_height])
        ax_ty    = self.fig.add_axes([left_margin, bottom_start + 0*slider_height, slider_width, slider_height])

        # Create sliders: adjust the ranges as needed for your images
        self.s_angle = Slider(ax_angle, 'Angle (°)', -180, 180, valinit=init_angle)
        self.s_tx = Slider(ax_tx, 'X Translation', -self.widefield.shape[1], self.widefield.shape[1], valinit=init_tx)
        self.s_ty = Slider(ax_ty, 'Y Translation', -self.widefield.shape[0], self.widefield.shape[0], valinit=init_ty)


        # Connect the sliders to the update function
        self.s_angle.on_changed(self.update)
        self.s_tx.on_changed(self.update)
        self.s_ty.on_changed(self.update)

        # Optionally, add a button to confirm or save the transformation
        ax_button = plt.axes([0.9, 0.5, 0.1, 0.04])
        button = Button(ax_button, 'Confirm', hovercolor='0.975')
        def on_confirm(event):
            print("Final parameters: Angle = {:.2f}°, X Translation = {:.2f}, Y Translation = {:.2f}".format(
                self.s_angle.val, self.s_tx.val, self.s_ty.val))
        button.on_clicked(on_confirm)

        plt.show()


    def atlas_rgba_mask(self, atlas:ndarray, threshold:float = .999)->ndarray:
        mask = atlas < threshold
        mask_thick = dilation(mask, disk(2))
        # Create an RGBA image initialized to 0
        rgba_atlas = np.zeros((atlas.shape[0], atlas.shape[1], 4), dtype=np.float32)
        
        # Choose a color for the outline (e.g., orange) and alpha=1
        rgba_atlas[mask, 0] = 1   # R
        rgba_atlas[mask, 1] = 0.5   # G
        rgba_atlas[mask, 2] = 0.0   # B
        rgba_atlas[mask, 3] = 1.0   # A

        return rgba_atlas
    

    def transform_atlas(self, atlas_img, angle, tx, ty, output_shape):
        """
        Applies a combined rotation and translation to the atlas image.
        
        Parameters:
        atlas_img: the input atlas image (as a NumPy array)
        angle: rotation angle in degrees
        tx, ty: translations along x and y
        
        Returns:
        The transformed atlas image.
        """
        # Create the affine transformation matrix - around the center.
        center = (atlas_img.shape[1] / 2, atlas_img.shape[0] /2)

        # Create the sequence of transforms (for center transform):
        # 1. Translate so the center becomes the origin
        tform_center = AffineTransform(translation=(-center[0], -center[1]))
        # 2. Rotate by the specified angle (convert to radians)
        tform_rot = AffineTransform(rotation=np.deg2rad(angle))
        # 3. Translate back to the original center
        tform_back = AffineTransform(translation=center)
        # 4. Apply additional translation (tx, ty)
        tform_trans = AffineTransform(translation=(tx, ty))
        
        # Compose the transforms: note that addition composes them.
        # The order is important: first shift center, rotate, shift back, then translate.
        tform_around_center = tform_center + tform_rot + tform_back + tform_trans
        transformed = warp(atlas_img, tform_around_center.inverse, preserve_range=True, output_shape=output_shape, mode = 'constant', cval = 1)

        # # Simple transform around top left corner:
        # tform = AffineTransform(rotation=np.deg2rad(angle), translation=(tx, ty))
        # # warp applies the inverse transformation to remap the output image
        # transformed = warp(atlas_img, tform.inverse, preserve_range=True, output_shape=output_shape, mode = 'constant', cval = 1)
        
        return transformed
    

    def update(self, val):
        """Callback function to update the overlay when a slider is changed."""
        angle = self.s_angle.val
        tx = self.s_tx.val
        ty = self.s_ty.val
        new_atlas = self.transform_atlas(self.atlas, angle, tx, ty, self.widefield.shape[:2])
        self.overlay.set_data(self.atlas_rgba_mask(new_atlas))
        self.fig.canvas.draw_idle()



class ABA_Aligner:
    def __init__(self):
        # Ask to choose folders
        # self.choose_folders()
        GUIres = AtlasWidefieldGUI()
        breakpoint()

    def choose_folders(self):
        print('Choose folders with widefield, two-photon and ABA mask and ABA XY coordinates to be aligned')
        parent_folders = []
        
        while True:
            chosen_dir = filedialog.askdirectory()
            if not chosen_dir: break
            parent_folders.append(chosen_dir)
        
        print(parent_folders)

if __name__ == '__main__':
    ABA_Aligner()