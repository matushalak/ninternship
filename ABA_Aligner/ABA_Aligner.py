from numpy import ndarray, array, deg2rad, fliplr, flipud, rot90
import numpy as np # eventually remove
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from matplotlib.widgets import Slider, Button
from skimage import io, exposure, filters, morphology
from skimage.transform import AffineTransform, warp, resize, hough_circle, hough_circle_peaks
import skimage as skim # eventually remove
from skimage.morphology import dilation, disk
from tkinter import filedialog
from SPSIG import SPSIG

class ImageLoader:
    def __init__(self, folder:str = ''):
        # Find image files
        # TODO
        # self.locate_files()

        # Load your images (update the file paths as needed)
        atlas_raw = io.imread('ABA-Mask.png', as_gray = True)          # brain atlas outlines image, 240.51 x 240.42 pt
        coords_raw = io.imread('examples/Eta/AllenBrainAtlasOverlay_Eta_CoordinatesXY.png')
        self.twophoton_raw = io.imread('examples/Eta/AllenBrainAtlasOverlay_Eta_Original.png', as_gray = True) # 1x zoom: 75,2019 x 75,2019 pt; 1.3x zoom: 57,8476 x 57,8476 pt; 1.6x zoom: 47 x 47 pt
        self.pRF = rot90(io.imread('examples/Eta/pRF_fieldsign.jpg'), k = -1).astype(float) / 255    # pRF image 330,9326 pt x 385,515 pt
        self.widefield = rot90(io.imread('examples/Eta/Brain.jpg', as_gray = True), k = -1).astype(float) # will modify later when preparing masks

        self.atlas = resize_image_pt(self.pRF, (385.515, 330.9326),
                                     atlas_raw, (240.42, 240.51))
        self.coords = resize_image_pt(self.pRF, (385.515, 330.9326),
                                      coords_raw, (240.42, 240.51))
        self.twop_mag = (57.8476, 57.8476) #self.find_2p_magnification()
        
        self.twophoton = resize_image_pt(self.pRF, (385.515, 330.9326),
                                         self.twophoton_raw, self.twop_mag)
            
    # scaling  = C / zoom; C = dimensions at 1x magnification
    def find_2p_magnification(zoom:float)->tuple[float, float]:
        return (75.2019 / zoom, 75.2019 / zoom)
    
    # TODO
    def locate_files(folder:str) -> tuple[str, str, str, str]:
        pass



class AtlasWidefieldGUI(ImageLoader):
    def __init__(self):
        super().__init__()
        # breakpoint()
        # Initial transformation parameters
        init_angle = 0       # in degrees
        init_tx = self.pRF.shape[1] // 2 - self.atlas.shape[1] // 2
        init_ty = self.pRF.shape[0] // 2 - self.atlas.shape[0] // 2

        # Create the figure and axes
        self.fig = plt.figure(figsize = (10,8))

        # Display the pRF image
        ax_main = self.fig.add_axes([0.05, 0.2, 0.9, 0.75])  # left=5%, bottom=20%, width=90%, height=75%
        ax_main.imshow(self.pRF)
        ax_main.set_title("Adjust Atlas Outlines")
        # ax_main.axis('off')

        # Apply initial transformation to the atlas outlines and overlay it
        atlas_transformed = self.transform_atlas(self.atlas, init_angle, init_tx, init_ty, self.pRF.shape[:2])
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
        self.s_tx = Slider(ax_tx, 'X Translation', -self.pRF.shape[1], self.pRF.shape[1], valinit=init_tx)
        self.s_ty = Slider(ax_ty, 'Y Translation', -self.pRF.shape[0], self.pRF.shape[0], valinit=init_ty)


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
        new_atlas = self.transform_atlas(self.atlas, angle, tx, ty, self.pRF.shape[:2])
        self.overlay.set_data(self.atlas_rgba_mask(new_atlas))
        self.fig.canvas.draw_idle()



class PrepareMasks:
    def __init__(self,
                 pRF:ndarray,
                 widef:ndarray,
                 twophoton:ndarray,
                 twop_in_points:tuple[float, float] = (75.2019, 75.2019)):
        '''
        main output of this class are .WF and ._2P binary masks already in correct scaling used for alignment
        '''
        # Load raw resized images
        self.pRF = skim.color.rgb2gray(pRF)
        rectangle_pRF_pixels = self.pRF.shape # will crop this later
        self.twop = twophoton
        self.widef = widef
        
        # crop pRF image
        self.pRF, xy_shift = self.crop_pRF(self.pRF)

        # Crop widefield to match cropped pRF
        self.widef = self.match_wf_pRF(self.widef, self.pRF)

        # Enhance local contrast -Contrast Limited Adaptive Histogram Equalization (CLAHE), needed before Frangi vesselness filter
        self.widef = exposure.equalize_adapthist(self.widef, clip_limit=0.05)
        self.twop = exposure.equalize_adapthist(self.twop, clip_limit=0.05)

        # Main function that extracts the Widefield and Two-photon masks to be aligned
        self.mask_2p, self.mask_wf = self.vessel_masks(self.widef, self.twop)

        # resize 2-photon mask appropriately
        self.mask_2p = resize_image_pt(reference_image=rectangle_pRF_pixels, reference_pt_dim=(389.0883, 334),
                                   target_image=self.mask_2p, target_pt_dim=twop_in_points, 
                                   AA = False) # Antialiasing must be false for boolean images
        

    def crop_pRF(self, pRF:ndarray)->ndarray:
        ''' crop away white areas surrounding (pRF) image'''
        # mask
        content_mask = pRF < .85
        content_coords = np.argwhere(content_mask) # indices of nonzero elements

        # coords.min(axis=0) and coords.max(axis=0) give the bounding box
        y0, x0 = content_coords.min(axis=0)
        y1, x1 = content_coords.max(axis=0) + 1  # +1 because slicing is exclusive on the upper bound

        return pRF[y0:y1, x0:x1], [x0, y0] # x, y translation to correct for whitespace crop

    def crop_wf(self, wf:ndarray)->ndarray:
        '''crop away axis labels and whitespace BURNED into (widefield) image...'''
        rows, cols = wf.shape
        for irow in range(rows):
            if wf[irow,:].mean() > .5:
                # print(f'Skip row {irow}')
                continue
            else:
                for icol in range(cols):
                    if wf[irow, icol:icol+100].mean() < .07 and wf[irow, icol] < 0.2:
                        # indexing cuts away axis labels
                        return self.crop_pRF(wf[irow:, icol:])[0] # crop_pRF cuts away leftover white on the sides
                
    def match_wf_pRF(self, wf:ndarray, pRF:ndarray)->ndarray:
        wf = self.crop_wf(wf)
        return resize(wf, pRF.shape, anti_aliasing = True)

    def vessel_masks(self, wf:ndarray, _2p:ndarray)->tuple[ndarray, ndarray]:
        # frangi vesselness filter
        f2p = filters.frangi(_2p, sigmas = np.arange(15, 60, 10))
        fwf = filters.frangi(wf, sigmas = np.arange(2, 20, 2))

        # Local thresholding, 
        # TODO: try LOCAL Otsu, should find optimal local
        widefield_thresh = filters.threshold_local(fwf, block_size = 3, offset = -0.0003)
        twop_thresh = filters.threshold_local(f2p, block_size = 17, offset = -0.00075)

        # TODO: More work / improvements - Morphological Operations: Closing gaps [dilation, erosion]; Skeletonization-Thinning (emphasize thin vessels since all vessels will be thing)
        # TODO: Fusion with original image & Multi-step filtering
        
        # Binary Masks
        mask_2p  = f2p > twop_thresh
        mask_wf = fwf > widefield_thresh

        return mask_2p, mask_wf



class ABA_Aligner:
    def __init__(self):
        # Ask to choose folders
        # self.choose_folders()
        GUIres = AtlasWidefieldGUI()
        Masks = PrepareMasks(pRF=GUIres.pRF, widef=GUIres.widefield, twophoton=GUIres.twophoton_raw)
        breakpoint()

    def choose_folders(self)->list[str]:
        print('Choose folders with pRF, two-photon and ABA mask and ABA XY coordinates to be aligned')
        parent_folders = []
        
        while True:
            chosen_dir = filedialog.askdirectory()
            if not chosen_dir: break
            parent_folders.append(chosen_dir)
        
        return parent_folders

    def align_masks(self):
        pass

def resize_image_pt(reference_image:ndarray|tuple[int, int], # can also provide pixel size of reference image here
                    reference_pt_dim:tuple[float, float],
                    target_image:ndarray, target_pt_dim:tuple[float, float], AA:bool = True)->ndarray:
        # ratio the images should be in
        ratio = array(target_pt_dim) / array(reference_pt_dim)
        
        reference_pix = reference_image if isinstance(reference_image, tuple) else reference_image.shape[:2]
        target_pix_goal = reference_pix * ratio
        
        return resize(target_image, output_shape=target_pix_goal, anti_aliasing=AA)

def show_me(*imgs:ndarray, color = 'gray'):
    n_imgs = len(imgs)
    if n_imgs == 1:
        imshow(imgs[0], cmap = color)
        plt.show()
        plt.close()
    elif n_imgs > 1 and n_imgs%2 == 1: # Odd
        fig, axes = plt.subplots(1, n_imgs, figsize = (4*n_imgs, 4))
        for ax, img in zip(axes, imgs):
            ax.imshow(img, cmap = color)
        plt.tight_layout()
        plt.show()
        plt.close()
    elif n_imgs % 2 == 0:
        cols = n_imgs // 2
        rows = 2
        fig, axes = plt.subplots(rows, cols, figsize = (4*cols, 4*2))
        axes = axes.flatten()
        for ax, img in zip(axes, imgs):
            ax.imshow(img, cmap = color)
        plt.tight_layout()
        plt.show()
        plt.close()


if __name__ == '__main__':
    ABA_Aligner()