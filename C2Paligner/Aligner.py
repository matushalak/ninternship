# @matushalak
from numpy import ndarray, array, deg2rad, fliplr, flipud, rot90
import numpy as np # eventually remove
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from matplotlib.widgets import Slider, Button
from skimage import io, exposure, filters, morphology
from skimage.transform import AffineTransform, warp, resize, rotate #,  hough_circle, hough_circle_peaks
import skimage as skim # eventually remove
from skimage.morphology import dilation, disk
from skimage.registration import phase_cross_correlation
from tkinter import filedialog
from src.SPSIG import SPSIG
import os
from glob import glob
from scipy.spatial import distance as sp_distance
import scipy.optimize as sp_optim
import multiprocessing as mp
from C2Paligner import ABAXY, ABAMASK, ABAOVERLAY, EXAMPLES

class ImageLoader:
    def __init__(self, folder:str = ''):
        # For development and checking with manual Adobe Illustrator "ground truth"
        self.coords_check_file, self.ZOOM = None, None

        # Find image files in specified folder
        prf_file, widefield_file, _2p_file = self.locate_files(folder)

        # atlas and coords provided with the ABA_Aligner package
        atlas_raw = io.imread(ABAMASK, as_gray = True)          # brain atlas outlines image, 240.51 x 240.42 pt
        coords = io.imread(ABAXY)
        
        # Load your image files 
        # necessary ingredients in each folder
        # self.twophoton_raw = io.imread('examples/Eta/AllenBrainAtlasOverlay_Eta_Original.png', as_gray = True) # 1x zoom: 75,2019 x 75,2019 pt; 1.3x zoom: 57,8476 x 57,8476 pt; 1.6x zoom: 47 x 47 pt
        # self.pRF = rot90(io.imread('examples/Eta/pRF_fieldsign.jpg'), k = -1).astype(float) / 255    # pRF image 330,9326 pt x 385,515 pt
        # self.widefield = rot90(io.imread('examples/Eta/Brain.jpg', as_gray = True), k = -1).astype(float) # will modify later when preparing masks
        self.twophoton_raw = io.imread(_2p_file, as_gray = True) # 1x zoom: 75,2019 x 75,2019 pt; 1.3x zoom: 57,8476 x 57,8476 pt; 1.6x zoom: 47 x 47 pt
        if 'original' not in _2p_file.lower():
            # rotate my_mousy image properly
            self.twophoton_raw = self.preprocess_2p(self.twophoton_raw)

        self.pRF = rot90(io.imread(prf_file), k = -1).astype(float) / 255    # pRF image 330,9326 pt x 385,515 pt
        self.widefield = rot90(io.imread(widefield_file, as_gray = True), k = -1).astype(float) # will modify later when preparing masks

        # Resize atlas and coords images loaded automatically
        # (389.0884, 334.0001)
        # (385.515, 330.9326)
        # NOTE: FIX resizing these high res images might be taking up a lot of the runtime
        self.atlas = resize_image_pt(self.pRF, (389.0884, 334.0001),
                                     atlas_raw, (240.42, 240.51))
        self.coords = resize_image_pt(self.pRF, (389.0884, 334.0001),
                                      coords, (240.42, 240.51))
        
        # Resize 2-photon images
        self.twop_mag = self.find_2p_magnification()
        
        # self.twophoton = resize_image_pt(self.pRF, (389.0884, 334.0001),
        #                                  self.twophoton_raw, self.twop_mag)
        
        # IF you have coords-check, resize to 2-p size to compare with output
        if self.coords_check_file is not None:
            coords_check = io.imread(self.coords_check_file)
            self.COORDS_CHECK = resize_image_pt(self.pRF, (389.0884, 334.0001),
                                                coords_check, self.twop_mag)
            
    def preprocess_2p(self, spsig_2p:ndarray) -> ndarray:
        return rot90(fliplr(spsig_2p))
    
    # scaling  = C / zoom; C = dimensions at 1x magnification
    def find_2p_magnification(self)->tuple[float, float]:
        return (75.2019 / self.ZOOM, 75.2019 / self.ZOOM)
    
    def locate_files(self, folder) -> tuple[str, str, str]:
        folder_files = os.listdir(folder)
        # TODO: generalize naming convention to user-defined settings
        assert any(file.lower().endswith('prf_fieldsign.jpg') for file in folder_files), 'Widefield PRF image is required!'
        assert any(file.lower().endswith('brain.jpg') for file in folder_files), 'Widefield brain image is required!'
        assert any(file.lower().endswith('mymousy_img.png') or file.lower().endswith('_original.png') for file in folder_files), '2-photon image is required!'

        twopOriginalfilefound = False
        for file in sorted(folder_files):
            if file.lower().endswith('prf_fieldsign.jpg'):
                prf_file = os.path.join(folder, file)

            if file.lower().endswith('brain.jpg'):
                widefield_file = os.path.join(folder, file)

            if file.lower().endswith('_original.png'):
                _2p_file = os.path.join(folder, file)
                twopOriginalfilefound = True

            if file.lower().endswith('_mymousy_img.png'):
                _2p_file = os.path.join(folder, file)

            if file.lower().endswith('_normcorr.mat'):
                magnifications = [1.0, 1.3, 1.6, 2.0, 2.5, 3.2, 4.0, 5.0, 6.3, 8.0, 10.1, 12.7, 16.0]
                mag_index = int(SPSIG(os.path.join(folder, file)).info.config.magnification)
                self.ZOOM = magnifications[mag_index-1]
            
            # Already exported from illustrator
            if file.lower().endswith('_coordinatesxy.png'):
                self.coords_check_file = os.path.join(folder, file)
        
        # no _normcorr.mat file
        if self.ZOOM is None:
            while True:
                try:
                    inpt = float(input("Couldn't find ..._normcorr.mat file in your folder, please specify your zoom level (float) manually here:   "))
                    break
                except ValueError:
                    print('You need to input a float for your zoom-level (eg. 2.5)')
            self.ZOOM = inpt

        return (prf_file, widefield_file, _2p_file)
            



class AtlasWidefieldGUI(ImageLoader):
    def __init__(self, folder:str):
        super().__init__(folder=folder)
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
        ax_main.set_title(f"Adjust Atlas Outlines for {folder}")
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
            plt.close()
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
                 twop_in_points:tuple[float, float]):
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

        # TODO: incorporate !!!
        # GETTING RID OF NEURONS XXX
        # WF
        show_me(self.widef, wfbres := morphology.black_tophat(self.widef, disk(8)), 
        # this seems already very good for WF
        wfst := filters.sato(1-wfbres, sigmas = np.arange(4, 8))) #with this potentially don't even need mask 

        # 2P
        show_me(bres2p := morphology.black_tophat(self.twop, disk(6)), 
        wres2p := morphology.white_tophat(self.twop, disk(6)),self.twop,  
        prog2p := self.twop - wres2p + bres2p)
        
        # for 2p frangi probably better, puts less stuff on the image
        fr2p = filters.frangi(prog2p, sigmas = np.arange(15, 60, 10))
        nn2p = morphology.closing(fr2p, disk(17)) 
        thresh2p = filters.apply_hysteresis_threshold(nn2p, 0.03,0.1)
        
        # also interesting 
        sat2p = filters.sato(prog2p, sigmas = np.arange(25, 30))
        op2p = morphology.opening(sat2p, disk(11))

        print('Implement BETTER FILTERS!!! & think about CLAHE ON vs BEFORE better filters')
        breakpoint()
        # Enhance local contrast -Contrast Limited Adaptive Histogram Equalization (CLAHE), needed before Frangi vesselness filter
        self.widef = exposure.equalize_adapthist(self.widef, clip_limit=0.05)
        self.twop = exposure.equalize_adapthist(self.twop, clip_limit=0.05)

        # Main function that extracts the Widefield and Two-photon masks to be aligned
        self.mask_2p, self.mask_wf = self.vessel_masks(self.widef, self.twop)
        # mask_thick = dilation(mask, disk(2)) # dilation for widefield image

        # resize 2-photon mask appropriately
        self.mask_2p = resize_image_pt(reference_image=rectangle_pRF_pixels, reference_pt_dim = (389.0884, 334.0001),
                                       target_image=self.mask_2p, target_pt_dim=twop_in_points, 
                                       AA = False if all(self.mask_2p in (0,1)) else True) # Antialiasing must be false for boolean images
        

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
            if wf[irow,:].mean() > .3:
                # print(f'Skip row {irow}')
                continue
            else:
                for icol in range(cols):
                    if wf[irow:,icol].mean() > .3:
                        continue

                    elif wf[irow, icol:icol+100].mean() < .3 and wf[irow, icol] < 0.2:
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
        self.folders = [os.path.join(EXAMPLES,'Eta')] #choose_folders() 

        GUIresults = []
        self.PREPARED_MASKs = []
        # Initial steps for which user is present
        # TODO: optimize speed of GUI & image loading (eg. setattrs ?)
        for fld in self.folders:
            # 1) Manual alignment on all images of interest from selected folders
            GUIresults.append(GUIres := AtlasWidefieldGUI(folder=fld))
            # 2) Automated vessel mask extraction (relatively quick so part of loop)
            self.PREPARED_MASKs.append(PrepareMasks(pRF=GUIres.pRF, 
                                                    widef=GUIres.widefield, 
                                                    twophoton=GUIres.twophoton_raw,
                                                    twop_in_points = GUIres.twop_mag))
        
        # 3) Parallelized registration algorithm
        parallel_args = self.prepare_parallel_registration_args()

        # loop for debugging
        for wf, tp in parallel_args:
            align_masks(wf, tp)
        
        # this would be parallelized over all the images selected
        # with mp.Pool(processes=mp.cpu_count()) as pool:
        #     AlignedRes = pool.starmap(align_masks, parallel_args)

    def prepare_parallel_registration_args(self,
                                           attr_images_used : dict[str:str, 
                                                                   str:str] = {'wf':'mask_wf',
                                                                               '2p':'mask_2p'}
                                          ) -> list[tuple[ndarray, ndarray]]:
        return [(getattr(Masks, attr_images_used['wf']),
                getattr(Masks, attr_images_used['2p']))
                for Masks in self.PREPARED_MASKs]   




# -------------------------------- REGISTRATION ALGORITHM --------------------------------------
#               (must consist of picklable functions to be parallelized)
# TODO: try transformations with rgba of 2-photon image
# TODO: use cross-correlation to find best translation for each rotation
# TODO: implement CMA-ES / other EA to optimize non-continuous function without gradient
# TODO: start in evenly-spaced different regions of search space (different islands etc. w migration)
        # NOTE: eg. split to 4 or 9 evenly-spaced regions with searching for transformations that will only take place there
        # regular migration between islands
        # once region much more promising than others, focus more islands on that region
# TODO: once converging on solution, finetune with gradient optimization scipy.minimize

def align_masks(mask_WF:ndarray, mask_2P:ndarray) -> tuple[float, float, float]:
    '''
    Main function that performs alignment using rigid transformations (rotation & translation) 
    on already properly scaled images

    input arguments: Images / Masks to align
        mask_WF : ndarray = large widefield image / mask onto which the small two-photon image is registered
        mask_2P : ndarray = small 2-photon image / mask to be registered onto large widefield image
    
    output: Parameters for the best found rigid transformation
        out : tuple(float, float, float) = (translation_x, translation_y, rotation_angle_deg)
    '''
    # to position the 2p image in "center", but actually it's top left
    wfy, wfx = mask_WF.shape
    twpy, twpx = mask_2P.shape
    ini_dx = wfx // 2 - twpx // 2
    ini_dy = wfy // 2 - twpy // 2
    
    # cover whole range
    rotation_range = (-45,45) # in degrees
    rr = np.arange(rotation_range[0], rotation_range[1]+1)
    
    dx_range = (- ini_dx, ini_dx) # roll axis 1
    xr = np.arange(dx_range[0], dx_range[1]+1)
    
    dy_range = (- ini_dy, ini_dx) # roll axis 0
    yr = np.arange(dy_range[0], dy_range[1]+1)

    ini_guess = [np.random.choice(xr),np.random.choice(yr),np.random.choice(rr)] 
    
    # optimization
    result = sp_optim.minimize(COST,ini_guess,
                               bounds = [dx_range, dy_range, rotation_range],
                               args=(mask_WF, mask_2P),
                               method='Nelder-Mead')
    
    breakpoint()


def COST(params:tuple[int, int, float], 
         WF:ndarray, TWOP:ndarray) -> float:
    dx, dy, theta = params
    wfy, wfx = WF.shape
    twpy, twpx = TWOP.shape
    # to position the 2p image in "center", but actually it's top left
    ini_dx = wfx // 2 - twpx // 2
    ini_dy = wfy // 2 - twpy // 2

    # CONTINUOUS translation, suited for gradient optimization
    # Create an affine transform (rotation + translation)
    transform = AffineTransform(rotation=np.deg2rad(theta),
                                translation=(ini_dx + dx, ini_dy + dy))
    # Warp the widefield image using continuous interpolation
    WF_transformed = warp(WF, inverse_map=transform.inverse, preserve_range=True)

    # DISCRETE translation apply transform inuitively (not continuous, can't be optimized)
    # WF_translation = np.roll(WF, shift = (ini_dy + dy, ini_dx + dx), axis = (0, 1))
    # WF_transformed = rotate(WF_translation, angle = theta)
    
    return metric(TWOP, WF_transformed[:twpy, :twpx])



# formulated as MINIMIZATION problem
def metric(im1:ndarray, im2:ndarray, mode = 'dice') -> float:
    assert im1.shape == im2.shape, 'Shapes of the 2 images must be identical dimensions (already AFTER transform was applied to widefield image)!'

    match mode:
        # Dice dissimilarity index
        case 'dice':
            return sp_distance.dice(im1.flatten(), im2.flatten())
        case 'jaccard':
            return sp_distance.jaccard(im1.flatten(), im2.flatten())

def rgba_2p_overlay():
    pass        
        

#%% ---------------------------Helper functions--------------------------
def resize_image_pt(reference_image:ndarray|tuple[int, int], # can also provide pixel size of reference image here
                    reference_pt_dim:tuple[float, float],
                    target_image:ndarray, target_pt_dim:tuple[float, float], AA:bool = True)->ndarray:
    # ratio the images should be in
    ratio = array(target_pt_dim) / array(reference_pt_dim)
    
    reference_pix = reference_image if isinstance(reference_image, tuple) else reference_image.shape[:2]
    target_pix_goal = reference_pix * ratio
    
    return resize(target_image, output_shape=target_pix_goal, anti_aliasing=AA)

def show_me(*imgs:ndarray, color = 'gray'):
    ''''
    Useful for debugging steps in the algorithm
    '''
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

def choose_folders()->list[str]:
    print('Choose (ROOT folders with child) folders containing: ',
    '\n 1) Widefield PRF image ...pRF_fieldsign.jpg ',
    '\n 2) Two-photon image ..._Original.png OR ..._myMousy_img.png ',
    '\n 3) Widefield image of same FOV as the PRF image but greyscale ...Brain.jpg',
    '\n Recommended (4) File containing magnification info..._normcorr.mat')
    root_dirs = []
    file_folders = []
    
    # get root folder / folders in which it will look for folders containing
    while True:
        chosen_dir = filedialog.askdirectory()
        if not chosen_dir: 
            break
        else:
            root_dirs.append(chosen_dir)
    
    look_for = '**/*pRF_fieldsign.jpg'
    look_for_all = ('prf_fieldsign.jpg', 'brain.jpg', 'mymousy_img.png', '_original.png')

    for root in root_dirs:
        for potential_folder in glob(os.path.join(root, look_for), recursive=True):
            potential_folder = os.path.dirname(potential_folder)
            yes_no = [any(file.lower().endswith(fn) for fn in look_for_all) for file in os.listdir(potential_folder)]

            if sum(yes_no) in (3, 4):
                file_folders.append(potential_folder)

    return file_folders

#%% Runs the script
if __name__ == '__main__':
    results = ABA_Aligner()
    print('Done!')