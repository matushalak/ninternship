import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import warp, AffineTransform
import os
import tkinter as tk
from tkinter import filedialog

def choose_folders():
    parent_folders = []
    
    while True:
        chosen_dir = filedialog.askdirectory()
        if not chosen_dir: break
        parent_folders.append(chosen_dir)
    
    print(parent_folders)
    # TODO: Continue from here

class AtlasTransformGUI:
    def __init__(self, widefield_path, atlas_path):
        # Load images
        self.widefield = io.imread(widefield_path)
        self.atlas = io.imread(atlas_path)
        self.atlas_h, self.atlas_w = self.atlas.shape[:2]
        
        # Initial transformation parameters:
        # rotation in radians and translations in pixels
        self.theta = 0.0  
        self.tx = 0.0
        self.ty = 0.0
        
        # Set the rotation center (the atlas image center)
        self.cx = self.atlas_w / 2
        self.cy = self.atlas_h / 2

        # Variables to keep track of interaction mode
        self.mode = 'none'  # can be 'none', 'translate', or 'rotate'
        self.start_mouse = None  # starting mouse (x,y) for the drag
        self.start_tx = None
        self.start_ty = None
        self.start_theta = None
        self.rotation_start_angle = None  # initial angle from center to mouse at start of rotation
        
        # Setup the figure and display images
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.widefield)
        self.ax.set_title('Drag inside atlas to translate, drag corners to rotate')
        
        # Display the transformed atlas overlay
        self.atlas_img_artist = self.ax.imshow(self.transformed_atlas(), alpha=0.5, cmap='hot')
        
        # Draw corner handles (will update on each transformation)
        self.handle_artists = []
        self.draw_handles()
        
        # Connect the interactive events
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
    
    def transformed_atlas(self):
        """
        Returns the atlas image transformed by the current rotation (about its center) and translation.
        The transformation matrix is:
           T = T(translation) · T(center) · R(theta) · T(-center)
        """
        # Compute the rotation matrix
        R = np.array([[np.cos(self.theta), -np.sin(self.theta)],
                      [np.sin(self.theta),  np.cos(self.theta)]])
        center = np.array([self.cx, self.cy])
        # The effective translation offset is: center - R @ center + [tx, ty]
        offset = center - R @ center + np.array([self.tx, self.ty])
        # Build the full 3x3 affine transformation matrix
        matrix = np.array([[np.cos(self.theta), -np.sin(self.theta), offset[0]],
                           [np.sin(self.theta),  np.cos(self.theta), offset[1]],
                           [0, 0, 1]])
        tform = AffineTransform(matrix=matrix)
        # Use warp with the inverse mapping
        transformed = warp(self.atlas, tform.inverse, preserve_range=True)
        return transformed

    def get_transformed_corners(self):
        """
        Computes the transformed positions of the atlas image's four corners.
        Corners are defined in the atlas's original coordinate system:
          (0,0), (width,0), (width,height), and (0,height).
        """
        corners = np.array([[0, 0],
                            [self.atlas_w, 0],
                            [self.atlas_w, self.atlas_h],
                            [0, self.atlas_h]])
        R = np.array([[np.cos(self.theta), -np.sin(self.theta)],
                      [np.sin(self.theta),  np.cos(self.theta)]])
        center = np.array([self.cx, self.cy])
        transformed_corners = []
        for corner in corners:
            new_corner = R @ (corner - center) + center + np.array([self.tx, self.ty])
            transformed_corners.append(new_corner)
        return np.array(transformed_corners)

    def draw_handles(self):
        """
        Draws markers on the corners of the transformed atlas.
        These serve as handles for initiating a rotation.
        """
        # Remove any previous handle markers
        for handle in self.handle_artists:
            handle.remove()
        self.handle_artists = []
        
        corners = self.get_transformed_corners()
        for corner in corners:
            # Draw a red circle for each corner
            handle, = self.ax.plot(corner[0], corner[1], 'ro', markersize=8)
            self.handle_artists.append(handle)
        self.fig.canvas.draw_idle()
    
    def get_handle_under_point(self, x, y, threshold=10):
        """
        Checks if the (x, y) mouse position is within a given threshold (pixels)
        of any corner handle. Returns the index of the handle if found, otherwise None.
        """
        corners = self.get_transformed_corners()
        for i, corner in enumerate(corners):
            if np.hypot(x - corner[0], y - corner[1]) < threshold:
                return i
        return None

    def on_press(self, event):
        """Handle mouse press events."""
        if event.inaxes != self.ax:
            return
        # First, check if the click is near one of the corner handles
        handle_idx = self.get_handle_under_point(event.xdata, event.ydata)
        if handle_idx is not None:
            self.mode = 'rotate'
            self.start_mouse = np.array([event.xdata, event.ydata])
            self.start_theta = self.theta
            center = np.array([self.cx, self.cy]) + np.array([self.tx, self.ty])
            # Compute the angle between the center and the mouse position
            self.rotation_start_angle = np.arctan2(self.start_mouse[1] - center[1],
                                                   self.start_mouse[0] - center[0])
        else:
            # If not near a handle, check if the click is inside the atlas region
            corners = self.get_transformed_corners()
            min_x, max_x = corners[:,0].min(), corners[:,0].max()
            min_y, max_y = corners[:,1].min(), corners[:,1].max()
            if min_x <= event.xdata <= max_x and min_y <= event.ydata <= max_y:
                self.mode = 'translate'
                self.start_mouse = np.array([event.xdata, event.ydata])
                self.start_tx = self.tx
                self.start_ty = self.ty
            else:
                self.mode = 'none'

    def on_motion(self, event):
        """Handle mouse motion events (dragging)."""
        if event.inaxes != self.ax or self.mode == 'none':
            return
        if self.mode == 'translate':
            # Compute displacement from the starting mouse position
            dx = event.xdata - self.start_mouse[0]
            dy = event.ydata - self.start_mouse[1]
            self.tx = self.start_tx + dx
            self.ty = self.start_ty + dy
        elif self.mode == 'rotate':
            # For rotation, compute the angle difference relative to the center
            center = np.array([self.cx, self.cy]) + np.array([self.tx, self.ty])
            current_mouse = np.array([event.xdata, event.ydata])
            current_angle = np.arctan2(current_mouse[1] - center[1],
                                       current_mouse[0] - center[0])
            d_angle = current_angle - self.rotation_start_angle
            self.theta = self.start_theta + d_angle
        
        # Update the atlas overlay and corner handles
        self.atlas_img_artist.set_data(self.transformed_atlas())
        self.draw_handles()
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        """Reset the mode when the mouse button is released."""
        self.mode = 'none'

    def show(self):
        plt.show()

if __name__ == '__main__':
    # choose_folders()
    # Usage:
    # Replace 'widefield.png' and 'atlas.png' with the paths to your images.
    gui = AtlasTransformGUI(widefield_path='examples/Beta/pRF_fieldsign.jpg', 
                            atlas_path='ABA-Mask.png')
    gui.show()