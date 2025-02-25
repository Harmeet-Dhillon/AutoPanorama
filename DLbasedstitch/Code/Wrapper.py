#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Code starts here:

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from helpers.data_gen import data_gen_main
# from Network.Network import \
#     TensorDLT  # for testing only! Remove after TensorDLT is implemented
from Network.Network import SupervisedNet
from helpers.stitch import *
# Add any python libraries here

def read_images(image_paths):
    """
    Read a set of images for Panorama stitching.
    """
    images = [cv2.imread(path) for path in image_paths]
    return images

def get_H4pt(model, img, supervised=True):
    """
    Obtain Homography using Deep Learning Model (Supervised and Unsupervised).
    """
    # # Convert images to tensor
    # patchA_tensor = torch.tensor(patchA, dtype=torch.float32).permute(2, 0, 1)
    # patchB_tensor = torch.tensor(patchB, dtype=torch.float32).permute(2, 0, 1)

    # stacked_tensor = np.concatenate((patchA_tensor, patchB_tensor),axis=-1)
    # stacked_tensor = np.transpose(stacked_tensor, (2, 0, 1))

    p = 32 # disturbance
    t = 0 # translation
    patch_size = 128

    stacked_patch_tensor, H4pt, cA, cB = data_gen_main(img, p, t, patch_size, verbose=True)
    stacked_patch_tensor = stacked_patch_tensor.reshape(1, 6, 128, 128)
    
    H4pt_pred = model(stacked_patch_tensor)

    return cA, cB, H4pt, H4pt_pred

# def compute_homography(H4pt, patch_size):
#     """
#     Compute the 3×3 homography matrix from the 4-point displacement H4pt.
#     """
#     # Define original patch corners (assume patch_size × patch_size)
#     h, w = patch_size
#     src_pts = np.array([[0, 0], [0, h], [w, 0], [w, h]], dtype=np.float32)  

#     # Apply H4pt offsets to get the destination points
#     dst_pts = src_pts + H4pt.reshape(4, 2)  # Reshape H4pt to 4×2

#     # Compute homography matrix
#     H, _ = cv2.findHomography(src_pts, dst_pts)
#     return H

# def warp_image(img, H, output_size):
#     """
#     Apply the homography matrix H to warp the image.
#     """
#     warped_img = cv2.warpPerspective(img, H, output_size, flags=cv2.INTER_LINEAR)
#     return warped_img

def get_H4pt_stitching(model, I1, I2, supervised=True):
    """
    Obtain Homography using Deep Learning Model (Supervised and Unsupervised).
    """
    If = np.concatenate((I1, I2), axis=-1)  # Concatenate along channel dimension
    If = np.transpose(If, (2, 0, 1))  # Change to (C, H, W)

    # Convert to tensor and add batch dimension
    stacked_patch_tensor = torch.from_numpy(If).float().unsqueeze(0)  # Shape: (1, 6, H, W)

    # Forward pass through model
    H4pt_pred = model(stacked_patch_tensor)

    return H4pt_pred



def drawPatches(img, cornersA, cornersB, cornersB_pred):  
    # Convert (row, col) -> (col, row) format (swap x and y)
    cornersA = np.array(cornersA, dtype=np.int32)[:, ::-1]  # Swap (y, x) -> (x, y)
    cornersB = np.array(cornersB, dtype=np.int32)[:, ::-1]  
    cornersB_pred = np.array(cornersB_pred, dtype=np.int32)[:, ::-1] 

    # Ensure correct ordering (top-left -> top-right -> bottom-right -> bottom-left)
    cornersA = cornersA[[0, 2, 3, 1]]  # Reordering
    cornersB = cornersB[[0, 2, 3, 1]] 
    cornersB_pred = cornersB_pred[[0, 2, 3, 1]] 

    # Convert the image from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create a figure and axis to plot the image and quadrilaterals
    fig, ax = plt.subplots(figsize=(8, 8))

    # Display the image using matplotlib in RGB color space
    ax.imshow(img_rgb)

    # Add quadrilaterals using matplotlib patches
    quadA = patches.Polygon(cornersA, closed=True, edgecolor='r', facecolor='none', linewidth=2, label="Corners A")
    quadB = patches.Polygon(cornersB, closed=True, edgecolor='b', facecolor='none', linewidth=2, label="Corners B")
    quadB_pred = patches.Polygon(cornersB_pred, closed=True, edgecolor='g', facecolor='none', linewidth=2, label="Predicted B")

    # Add the quadrilaterals to the plot
    ax.add_patch(quadA)
    ax.add_patch(quadB)
    ax.add_patch(quadB_pred)

    # Function to find the top-right corner (max x, min y)
    def top_right_corner(corners):
        return corners[np.argmax(corners[:, 0]), :]  # max x, and corresponding y

    # Get top-right corners of each quadrilateral
    top_right_A = top_right_corner(cornersA)
    top_right_B = top_right_corner(cornersB)
    top_right_B_pred = top_right_corner(cornersB_pred)

    # Add text labels at the top-right corners of the quadrilaterals
    ax.text(top_right_A[0], top_right_A[1], 'Corners A', color='r', fontsize=12, ha='left', va='bottom')
    ax.text(top_right_B[0], top_right_B[1], 'Corners B', color='b', fontsize=12, ha='left', va='bottom')
    ax.text(top_right_B_pred[0], top_right_B_pred[1], 'Predicted B', color='g', fontsize=12, ha='left', va='bottom')

    # Turn off axis to just show the image and quadrilaterals
    ax.axis('off')

    # Display the plot
    plt.show() # comment out for debugging



def main(stitch=False):
    if stitch:
        BasePath = "../Data/Test"
        LabelPath = "./helpers"
        PatchPath = os.path.join(BasePath, "Patches")
        os.makedirs(PatchPath, exist_ok=True)

        checkpoint_path = "../Data/Checkpoints/105model.ckpt"
        model = SupervisedNet()
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Get only numerically named images (ignoring stitched results)
        image_files = sorted(
            [f for f in os.listdir(BasePath) if f.endswith('.jpg') and f.split('.')[0].isdigit()],
            key=lambda x: int(x.split('.')[0])  # Sort numerically
        )

        if len(image_files) < 2:
            print("Not enough images for stitching.")
            return

        # Start with the first numeric image
        stitched_img_path = os.path.join(BasePath, image_files[0])

        for i in range(1, len(image_files)):
            imgPath1 = stitched_img_path
            imgPath2 = os.path.join(BasePath, image_files[i])

            print(f"Processing: {imgPath1} + {imgPath2}")

            # Extract patches
            img1_corners, img2_corners = common_patches(imgPath1, imgPath2, BasePath)
            patch_Path1 = os.path.join(PatchPath, "patchB.jpg")
            patch_Path2 = os.path.join(PatchPath, "patchA.jpg")

            img1 = np.float32(cv2.imread(patch_Path1))
            img2 = np.float32(cv2.imread(patch_Path2))

            # Predict homography
            H4pt_pred = get_H4pt_stitching(model, img1, img2)
            H4pt_pred_np = H4pt_pred.detach().cpu().numpy().flatten()
            H4pt_pred_str = " ".join(map(str, H4pt_pred_np))

            # Save H4pt labels
            labels_file = os.path.join(LabelPath, "H4pt_labels.txt")
            with open(labels_file, "w") as f:
                f.write(H4pt_pred_str + "\n")

            print(f"H4pt labels saved to {labels_file}")

            # Compute final homography and stitch
            Hpath = labels_file
            h = get_Hfinal(Hpath, img1_corners, img2_corners)

            # Update stitched image path
            stitched_img_path = os.path.join(BasePath, f"stitch_{i}.jpg")
            stitch_operation(imgPath1, imgPath2, h, stitched_img_path)

            # Append stitched image for further stitching
            image_files.append(f"stitch_{i}.jpg")

        print(f"Final stitched image: {stitched_img_path}")

    else:
        # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

        """
        Read a set of images for Panorama stitching
        """

        BasePath = "../Data/Val/"

        img_idx = 6
        imgPath = BasePath + f"{img_idx}.jpg"
        img = cv2.imread(imgPath)
        """
        Obtain Homography using Deep Learning Model (Supervised and Unsupervised)
        """
        checkpoint_path = "../Data/Checkpoints/105model.ckpt"
        model = SupervisedNet()
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))  # Load checkpoint
        model.load_state_dict(checkpoint["model_state_dict"])  # Extract only model state
        model.eval()  # Set to evaluation mode

        cA, cB, H4pt, H4pt_pred = get_H4pt(model, img)
        H4pt_reconstructed = H4pt_pred.reshape(4, 2)

        print("H4pt: ", H4pt)
        print("H4pt_pred:", H4pt_reconstructed)

        cB_pred = H4pt_reconstructed.detach().numpy() + cA
        drawPatches(img, cA, cB, cB_pred)

        """TESTING ONLY. REMOVE AFTER TENSORDLT IS IMPLEMENTED"""

        # model = UnsupervisedNet()
        # warpedPatchA = getWarpedPA(model, img).squeeze(0)  # Remove batch dimension → [3, 128, 128]

        # # Convert to NumPy and rearrange dimensions from (C, H, W) to (H, W, C)
        # img_np = warpedPatchA.permute(1, 2, 0).cpu().numpy()

        # # Display the image
        # plt.imshow(img_np)
        # plt.axis("off")  # Hide axes
        # plt.show()

        """TESTING ONLY. REMOVE AFTER TENSORDLT IS IMPLEMENTED"""

        """
        Image Warping + Blending
        Save Panorama output as mypano.png
        """     
if __name__ == "__main__":
    main(stitch=True)

