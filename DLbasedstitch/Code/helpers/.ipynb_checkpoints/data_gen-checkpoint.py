
import argparse
import glob
import os
import random as rand

import cv2
import numpy as np
import torch
from tqdm import tqdm


def select_patchA(img,p,t,patch_size): # p = disturbance , t = translation
    row,col=img.shape[:2]
    #get the good coords for corner A ....assume corners of patch as ABCD 
    clearance=int(1.5*(p+t))
    colA=rand.randint(clearance,col-clearance-patch_size)
    rowA=rand.randint(clearance,row-clearance-patch_size)
    colB=colA+patch_size
    rowB=rowA
    colC=colA
    rowC=rowA+patch_size
    colD=colB
    rowD=rowC
    cornersA=[[rowA,colA],[rowB,colB],[rowC,colC],[rowD,colD],]
    # print("cornersA",cornersA)
    return cornersA

def disturbed_patch(img,corners,p):
    cornersB=[]
    for i,(row,col) in enumerate(corners):
        row=row+rand.randint(-p,p)
        col=col+rand.randint(-p,p)
        corner=[row,col]
        cornersB.append(corner)
    #print(" disturbed corners",cornersB)
    return cornersB

def rowcol_to_colrow(list1):
    return [[col, row] for row, col in list1]

def get_homography(img,cornersA,cornersB):
    #getperspective takes input in form of x,y /col,row
    h=cv2.getPerspectiveTransform(np.float32(rowcol_to_colrow(cornersA)), np.float32(rowcol_to_colrow(cornersB))) ###i m not sure about it why not this...
    h=np.linalg.pinv(h)
    ###will try doing this manually
    #print('h',h)
    return h

def warp_img(img,h):
    # Get dimensions of img1 and img2
    #this function perspective.transform also takes input in form of col,row
    h1, w1 = img.shape[:2]

    # Compute the warped corners of img1
    '''
    corners = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]], dtype=np.float32)
    warped_corners = cv2.perspectiveTransform(corners[None, :, :], h)[0]
    # Find the bounding box of the warped corners
    x_min = warped_corners[:, 0].min()
    y_min = warped_corners[:, 1].min()
    x_max = warped_corners[:, 0].max()
    y_max = warped_corners[:, 1].max()

    # Compute the size of the new canvas
    new_width = int(x_max - x_min)
    new_height = int(y_max - y_min)
    '''
    # Compute the translation matrix to shift the content to fit in the canvas
    #translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

    # Warp img1 onto the extended canvas
    #extended_h = translation @ h  # Combine translation with the homography
    warp_img = cv2.warpPerspective(img,h, (img.shape[1],img.shape[0]))
    return warp_img

def patch_extraction(img, corners):  # corners in form of [row, col]
    row_min = corners[0][0]
    col_min = corners[0][1]
    row_max = corners[3][0]  # Include the max row
    col_max = corners[3][1] # Include the max col
    patch_img = img[row_min:row_max, col_min:col_max]  # Correct slicing
    return patch_img

def data_factory(img,p,t,patch_size):
    cornersA=select_patchA(img,p,t,patch_size)
    cornersB=disturbed_patch(img,cornersA,p)
    
    cA = np.array(cornersA, dtype=np.float32)  # Ensure float32
    cB = np.array(cornersB, dtype=np.float32)  # Ensure float32
    
    H4pt=cB-cA
    h=get_homography(img,cornersA,cornersB)
    warped_img=warp_img(img,h)
    img1=patch_extraction(img,cornersA)
    img2=patch_extraction(warped_img,cornersA)
    return img1,img2,H4pt

def data_gen_main(img, p, t, patch_size):
    img1, img2 = data_factory(img, p, t, patch_size)

    # Convert images to PyTorch tensors
    img1_tensor = torch.tensor(img1, dtype=torch.float32).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    img2_tensor = torch.tensor(img2, dtype=torch.float32).permute(2, 0, 1)  

    # Stack tensors into shape (2, 3, patch_height, patch_width)
    final_tensor = torch.stack([img1_tensor, img2_tensor], dim=0)  

    return final_tensor

def generate_data(src_imgs_path, dst_imgs_path, dst_txt_path):
    """
    dst_imgs_path: Path to folder containing images to extract and create patches from
    dst_txt_path: Path to folder to store txt labels
    """

    # Create directories for storing results
    os.makedirs(f"{dst_imgs_path}/A", exist_ok=True)
    os.makedirs(f"{dst_imgs_path}/B", exist_ok=True)
    os.makedirs(dst_txt_path, exist_ok=True)

    # Paths to text files
    labels_file = f"{dst_txt_path}/Labels.txt"
    dirA_file   = f"{dst_txt_path}/DirNamesA.txt"
    dirB_file   = f"{dst_txt_path}/DirNamesB.txt"

    # Ensure the text files exist
    if not os.path.exists(labels_file):
        open(labels_file, 'w').close()
    if not os.path.exists(dirA_file):
        open(dirA_file, 'w').close()
    if not os.path.exists(dirB_file):
        open(dirB_file, 'w').close()

    # Get all images from the source folder
    # img_files = glob.glob("./Data/Train/*.jpg")
    img_files = glob.glob(src_imgs_path+"/*.jpg")

    # Sort the list of image files to ensure ordered processing
    img_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))

    p = 32 # disturbance
    t = 0 # translation
    patch_size = 128

    # Open label and DirNames files for writing
    with open(labels_file, "w") as f_labels, open(dirA_file, "w") as f_A, open(dirB_file, "w") as f_B:
        print("Opened files")
        for img_path in tqdm(img_files, desc="Processing Images"):
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error loading {img_path}")
                continue

            h, w = img.shape[:2]
            if min(h, w) < 240:
                p = min(h, w) // 25
                # print("p", p)

            img_name = os.path.basename(img_path).split('.')[0]

            # Process image to get patches and homography
            img1, img2, H4pt = data_factory(img, p, t, patch_size)

            # Save img1 in folder A and img2 in folder B
            cv2.imwrite(f"{dst_imgs_path}/A/{img_name}.jpg", img1)
            cv2.imwrite(f"{dst_imgs_path}/B/{img_name}.jpg", img2)

            # Write the paths to DirNamesTrainA and DirNamesTrainB without the .jpg extension
            f_A.write(f"{dst_imgs_path}/A/{img_name}\n")
            f_B.write(f"{dst_imgs_path}/B/{img_name}\n")

            # Convert H4pt from 4x2 to a single-row array (1x8)
            H4pt_flat = H4pt.flatten()
            H4pt_str = " ".join(map(str, H4pt_flat))  # Convert array to space-separated string

            # Save the labels in the text file
            f_labels.write(f"{img_name} {H4pt_str}\n")  



def main():

    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--DataDir",
        default="",
        help="Base path of images, Default:../../Data/",
    )

    Args = Parser.parse_args()
    DataDir = Args.DataDir

    # If DataDir is empty, set it to the grandparent directory of the script
    if DataDir == "":
        DataDir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    # Training
    print("Creating training data")

    train_src_images = os.path.join(DataDir, "Data/Train")
    train_dst_images_path = os.path.join(DataDir, "Data/Train/Patches")
    train_dst_text_path = os.path.join(DataDir, "Data/Train/TxtFiles")

    generate_data(train_src_images, train_dst_images_path, train_dst_text_path)

    # Validation
    print("Creating validation data")

    val_src_images = os.path.join(DataDir, "Data/Val")
    val_dst_images_path = os.path.join(DataDir, "Data/Val/Patches")
    val_dst_text_path = os.path.join(DataDir, "Data/Val/TxtFiles")

    generate_data(val_src_images, val_dst_images_path, val_dst_text_path)

if __name__ == "__main__":
    main()



