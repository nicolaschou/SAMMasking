"""
========================================
File:        sam_point_mask.py
Description: This script masks objects using the Segment Anything Model 
             (SAM) Predictor and user-selected reference points.

Citation:
    @article{kirillov2023segany,
    title={Segment Anything},
    author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and 
            Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and 
            Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. 
            and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
    journal={arXiv:2304.02643},
    year={2023}
    }
             
Author:      Nico Chou
Created:     7/18/2025
Last Edited: 8/13/2025
========================================
"""


import torch
import segment_anything
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
from mpl_interactions import zoom_factory
import numpy as np
import cv2
import os
import tkinter as tk
from tkinter import filedialog


class ImageData:
    """
    A container class for storing an image and its filename.

    Attributes:
        image (np.ndarray): The image data as a numpy array.
        filename (str): The name of the image file.
    """
    def __init__(self, image: np.ndarray, filename: str):
        """
        Initializes an ImageData object.

        Args:
            image (np.ndarray): The image data as a numpy array.
            filename (str): The name of the image file.
        """
        self.image = image
        self.filename = filename


def load_images() -> list:
    """
    Opens a file dialog for the user to select multiple image files and 
    loads the images as ImageData objects into a list.

    Returns:
        list: A list of ImageData objects containing the images in RGB 
              format selected by the user.
    """
    # Create a hidden root window
    root = tk.Tk()
    root.withdraw()

    # Ask user to select multiple files
    file_paths = list(filedialog.askopenfilenames(title="Select image files"))
    root.update()
    root.destroy()

    # Extract images from their paths
    images = []
    for file_path in file_paths:
        image = cv2.imread(file_path)
        if image is None:
            print(f"Warning: Could not read {file_path}")
            continue
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        filename = os.path.basename(file_path)
        images.append(ImageData(image, filename))

    if not images:
        print(
            "No valid images were loaded. Please check your files and try "
            "again."
        )

    return images


def get_reference_points(
        image_data: ImageData, 
        scale=0.3
    ) -> np.ndarray:
    """
    Displays an image and collects (x, y) points clicked by the user.

    Args:
        image_data (ImageData): ImageData object for the image.
        scale (float): How large the displayed image is relative to the 
                       original.

    Returns:
        np.ndarray: Array with clicked (x, y) coordinates in the 
        original dimensions.
    """
    # Plot resized image with zoom functionality
    image_scaled = cv2.resize(image_data.image, None, fx=scale, fy=scale)
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.07)
    ax.imshow(image_scaled)
    zoom_factory(ax)
    ax.set_title(
        f"Click reference points on the object(s) to mask "
        f"({image_data.filename})"
    )
    
    points = [] # stores keypoint coordinates
    dots = [] # stores the Line2D dot objects

    # Mouse click callback
    def onclick(event):
        # If a tool is selected, do nothing
        toolbar = event.canvas.toolbar
        if toolbar is not None and toolbar.mode != '':
            return
        
        # Plot dot at last mouse position when no tool is selected
        if event.xdata is not None and event.ydata is not None:
            x, y = event.xdata, event.ydata
            points.append((x, y))
            dot, = ax.plot(x, y, 'ro', ms=3) # unpack
            dots.append(dot)
            fig.canvas.draw()

    # Key press callback
    def onkey(event):
        # backspace -> remove last clicked point
        if event.key == 'backspace':
            if dots:
                dot = dots.pop()
                dot.remove()
                points.pop()
                fig.canvas.draw()
        
        # enter -> close figure
        if event.key == 'enter':
            if points:
                fig.canvas.mpl_disconnect(cid_click)
                fig.canvas.mpl_disconnect(cid_key)
                plt.close(fig)
            else:
                print("Please click a point first.\n")

    # Connect the event handler and display the image
    cid_click = fig.canvas.mpl_connect('button_press_event', onclick)
    cid_key = fig.canvas.mpl_connect('key_press_event', onkey)
    plt.show()

    # Wait until window is closed to return
    while plt.fignum_exists(fig.number):
        plt.pause(0.1)
    orig_points = np.array(points) / scale
    return orig_points


def generate_mask(
        image: ImageData, 
        reference_points: np.ndarray, 
        predictor: SamPredictor,
        file_prefix: str
    ) -> ImageData:
    """
    Generates a segmentation mask for the specified image using the 
    provided Segment Anything predictor and reference points.

    Args:
        image (ImageData): The input image and filename.
        reference_points (array-like): Coordinates of reference points. 
                                       All points are assumed to be in 
                                       the foreground.
        predictor (SamPredictor): A configured Segment Anything 
                                  predictor used for mask generation.
        file_prefix (str): Prefix to prepend to the output mask's 
                           filename.

    Returns:
        ImageData: A new ImageData object containing the generated 
        binary mask and prefixed filename
    """
    # Generate mask (assumes all points are in foreground)
    predictor.set_image(image.image)
    masks, scores, logits = predictor.predict(
        point_coords=reference_points,
        point_labels=np.ones(len(reference_points)),
        multimask_output=False
    )

    # Extract mask and return
    mask = masks[0]
    mask_uint8 = (mask * 255).astype(np.uint8)
    return ImageData(mask_uint8, file_prefix+image.filename)


def select_device() -> str:
    """
    Determines the most suitable PyTorch device for computation.

    Returns:
        str: The name of the selected device ("cuda", "mps", or "cpu").
    """
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def export_image(image_data: ImageData):
    """
    Saves the image and filename specified in an ImageData object to 
    disk.

    Args:
        image_data (ImageData): ImageData object containing the image.
    """
    cv2.imwrite(image_data.filename, image_data.image)


def mask_stack() -> list:
    """
    Loads images, collects reference points, and generates segmentation 
    masks for each image using the Segment Anything model.

    Returns:
        list: A list of ImageData objects containing the generated 
              masks. Returns None if the user chooses to quit before 
              processing.
    """
    images = load_images()

    # Get a filename prefix from the user
    invalid_chars = set('<>:"/\\|?*')
    while True:
        file_prefix = input(
            'Enter the output mask filename prefix or type "quit" to '
            'end the program:\n')
        if any(char in invalid_chars for char in file_prefix):
            print(
                f"Prefix contains invalid filename characters: "
                f"{''.join(invalid_chars)}. Try again.")
        elif file_prefix.strip() == "quit":
            return
        else:
            break

    # Collect reference points
    reference_points = []
    for image in images:
        reference_points.append(get_reference_points(image))

    # Set up SAM
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = select_device()
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # Generate a mask for each image with the reference points
    masks = []
    for i in range(len(images)):
        mask = generate_mask(
            images[i], 
            reference_points[i], 
            predictor, 
            file_prefix
        )
        export_image(mask)
        masks.append(mask)

    return masks


def main():
    mask_stack()


if __name__ == "__main__":
    main()