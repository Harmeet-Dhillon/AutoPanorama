import numpy as np
import cv2
import matplotlib.pyplot as plt
import random as rand
import os

def get_harris_corners(img, show=False):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    
    #result is dilated for marking the corners, not important - do we need this?
    dst_dilated = cv2.dilate(dst,None)

    if show:

        # Threshold for an optimal value, it may vary depending on the image - do we need this?
        # dst_dilated>0.01*dst_dilated.max() filters out the points
        # [0, 0, 255] marks the filtered points as red
        img[dst_dilated>0.01*dst_dilated.max()]=[0,0,255] 

        cv2.imshow('dst',img)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()

    #dst_dilated = dst_dilated[dst_dilated>0.01*dst_dilated.max()]
    img_copy=img.copy()
    img_copy[dst_dilated>0.01*dst_dilated.max()]=[255,0,0]
    
    # plt.figure(figsize=(5,5))
    # plt.imshow(img_copy)
    # plt.title('raw corners')
    # #plt.savefig("./YourDirectoryID_p1/Phase1/Data/Train/mid_outputs/harriscorners_image.jpg")
    # plt.show()
    return dst

def max_in_local(local_mat, threshval):
    max_val = np.max(local_mat)
    if max_val >= threshval:
        rows, cols = local_mat.shape
        #operation with Time o(n) and space o(1)
        for i in range(rows):
            for j in range(cols):
                if local_mat[i, j] == max_val:
                    return i, j
    return None 

def imgregionalmax(dst,patch_size):
    '''
    This function identifies all pixels that are local maxima in their neighborhood
    (determined by patch_size) and returns a dictionary with (row, col)-pixel_intensity
    key-value pairs, where the pixel_intensity is the corner-detection value as given in 
    dst

    inputs:
        dst: Output of image after applying corner detection (should be a 2D array)

    outputs:
        map_cimg: dictionary with keys as (row, col) cooridnate of a pixel and 
        value as corner detection output for that pixel
    '''
    #first get a threshold to ensure that the pixel previously qualified as corner
    thresh_dst=0.01*dst.max()
    #create a hash map of local maxima coords and its dst
    #choose a patch of patch_size just to take out numpy
    rows,cols = dst.shape
    t_row=0
    map_cimg={}
    while t_row<rows:
        t_col=0
        while t_col<cols:
            row_start=t_row
            row_end=min(t_row+patch_size,rows)
            col_start=t_col
            col_end=min(t_col+patch_size,cols)
            tmp_array=dst[row_start:row_end,col_start:col_end]
            local_coords=max_in_local(tmp_array,thresh_dst)
            if local_coords:
                global_coords=(row_start+local_coords[0],col_start+local_coords[1])
                map_cimg[global_coords]=dst[global_coords]
            t_col+=patch_size
        t_row+=patch_size
              
    return map_cimg



def ANMS(img,dst, n_best):
    """

    This function identifies "n_best" keypoints that are true local maxima for
    corner detection 

    inputs:
        dst: Output of image after applying corner detection (should be a 2D array)
    
    Returns: 
        best_n_coordinates: list of tuple (row, col) values of n_best keypoints
    """

    patch_size = 5
    map_cimg = imgregionalmax(dst, patch_size)
    map_r = {}
    print("[ANMS] Total number of local maxima keypoints",len(map_cimg))
    
    # Calculate the minimum distance for each keypoint
    for key1 in map_cimg:
        r = np.inf
        for key2 in map_cimg:
            if key1 != key2:
                ed = np.inf
                if map_cimg[key2] > map_cimg[key1]:
                    ed = (key1[1] - key2[1]) ** 2 + (key1[0] - key2[0]) ** 2
                if ed < r:
                    r = ed
        map_r[key1] = r
    
    # Sort the coordinates based on their score in descending order
    sorted_coordinates = sorted(map_r.items(), key=lambda item: item[1], reverse=True)
    
    # Return the best n coordinates
    best_n_coordinates = [coord for coord, _ in sorted_coordinates[:n_best]]
    ###added to save image ###
    
    
    img_copy = img.copy()  
   
    for (row, col) in best_n_coordinates:
        if 0 <= row < img_copy.shape[0] and 0 <= col < img_copy.shape[1]:  # Ensure within bounds
            cv2.circle(img_copy, (col, row), 3, (255, 0, 0), -1)  # Red circle (BGR format)
    
    
    
    # plt.figure(figsize=(10,5))
    # plt.imshow(img_copy)
    # plt.title('anms_image')
    # #plt.savefig("./YourDirectoryID_p1/Phase1/Data/Train/mid_outputs/anms_image.jpg")
    # plt.show()
    
    #print('best cordinates',best_n_coordinates)
    return best_n_coordinates

def get_feature_descriptors(img, feature_points):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    feature_descriptors = []

    for keypoint in feature_points:
        (row, col) = keypoint

        patch_length = 41
        half_patch_length = patch_length//2

        # Numpy automatically stops at last row/col if kernel size exceeds image bounds
        roi = gray[max(0, row - half_patch_length):row + half_patch_length + 1,
                max(0, col - half_patch_length):col + half_patch_length + 1]

        feature_descriptor = cv2.GaussianBlur(roi, ksize=(41, 41), sigmaX=0)

        subsampled_feature_descriptor =  cv2.resize(feature_descriptor, (8, 8))

        feature_vector = subsampled_feature_descriptor.flatten()

        standardized_feature_vector = (feature_vector - np.mean(feature_vector)) / np.std(feature_vector)

        feature_descriptors.append(standardized_feature_vector)

    return feature_descriptors


def match_features(feature_descriptors_1 ,feature_descriptors_2, ratio=0.5):
    '''
    inputs:
        feature_descriptors_1: List of [64x1] features for n keypoints in image 1
        feature_descriptors_1: List of [64x1] features for n keypoints in image 2
    outputs:
        matching_points: List of n cv2.DMatch<keypoint_1_id, keypoint_2_id, distance>
                         objects, where keypoint_1_id and keypoint_2_id are indices
                         of keypoints as indexed in feature_descriptors_1 and
                         feature_descriptors_2 respectively, and distance is the
                         sum-of-squared differences distance between feature
                         descriptors of these keypoints.
    '''

    print("[Match Features] Feature descriptors 1 length: ", len(feature_descriptors_1))
    print("[Match Features] Feature descriptors 2 length: ", len(feature_descriptors_2))
    assert len(feature_descriptors_1) == len(feature_descriptors_2), "Feature descriptor lengths should match!"
    
    matching_points = []

    for idx1, feature_1 in enumerate(feature_descriptors_1):
        # Create placeholder values for top 2 best matches for keypoint 1
        # This array list should be sorted in ascending order by DMatch.distance
        top_2_matches = [cv2.DMatch(_queryIdx=-1, _trainIdx=-1, _distance=np.inf-1.0), 
                                   cv2.DMatch(_queryIdx=-1, _trainIdx=-1, _distance=np.inf)]
        
        # Go through features for all keypoints to find the top 2 matches
        for idx2, feature_2 in enumerate(feature_descriptors_2):
            if idx1 != idx2:

                # Calculate sum of squared difference between features of keypoints
                distance = np.sum((feature_1 - feature_2)**2)

                # First check if calculated distance is less than 1st saved match distance
                if distance < top_2_matches[0].distance:

                    # Replace 2nd saved match with 1st saved match
                    top_2_matches[1] = top_2_matches[0]

                    # Replace 1st saved match with new distance
                    top_2_matches[0] = cv2.DMatch(_queryIdx=idx1, _trainIdx=idx2, _distance=distance)

                elif distance > top_2_matches[0].distance and distance < top_2_matches[1].distance:
                # If calculated distance is greater than first saved distance 
                # but smaller than second saved distance
                    top_2_matches[1] = top_2_matches[0]

        
        # Add top match if ration of distances between 1st and 2nd match < 0.5
        if top_2_matches[0].distance/top_2_matches[1].distance < 0.5:
            matching_points.append(top_2_matches[0])

    return matching_points

def rowcol_to_colrow(tuple_list):
    """
    Switch (row, col) tuples to (col, row).

    Args:
        tuple_list (list): List of (row, col) tuples.
    
    Returns:
        list: List of (col, row) tuples.

    # Example usage
    tuple_list = [(1, 2), (3, 4), (5, 6)]
    switched = switch_row_col(tuple_list)
    print(switched)  # Output: [(2, 1), (4, 3), (6, 5)]
    
    """
    return [(col, row) for row, col in tuple_list]

def crop_dimensions(ANMS): # we get a tuple of x,y 
    row_min=min(row for row,col in ANMS)
    col_min=min(col for row,col in ANMS)
    row_max=max(row for row,col in ANMS)
    col_max=max(col for row,col in ANMS)
    return row_min,row_max,col_min,col_max

        

def extreme_points(all_matches, patch_dims):
    # Extract patch dimensions
    rmin, rmax, cmin, cmax = patch_dims
    
    # Define the four corner points
    corners = [(rmin, cmin), (rmin, cmax), (rmax, cmin), (rmax, cmax)]
    
    # Filter matches within the valid patch region
    matches = [(row, col) for row, col in all_matches if rmin <= row <= rmax and cmin <= col <= cmax]
    
    matches = np.array(matches)

    # To store the four selected unique points
    selected_points = []

    # Find the closest unique points for each corner
    for corner in corners:
        dists = np.linalg.norm(matches - np.array(corner), axis=1)  # Compute Euclidean distances
        sorted_indices = np.argsort(dists)  # Sort indices by distance

        # Pick the first unique available point
        for idx in sorted_indices:
            candidate = tuple(matches[idx])
            if candidate not in selected_points:
                selected_points.append(candidate)
                break  # Move to the next corner

    # Ensure exactly 4 points are returned, handling duplicates
    # remaining_points = [tuple(p) for p in matches if tuple(p) not in selected_points]
    # selected_points.extend(remaining_points[: max(0, 4 - len(selected_points))])

    return selected_points


def extract_croppedimage(matches, dst):
    # Assumed matches as a list of keypoints [(row, col), ...]
    sorted_patches = {}
    edge =240

    while edge <= 240:
        row_min, row_max, col_min, col_max = crop_dimensions(matches)
        curr_row_min, curr_col_min = row_min, col_min

        while curr_row_min <= row_max - edge:
            while curr_col_min <= col_max - edge:
                for row, col in matches:
                    if curr_row_min <= row < curr_row_min + edge and curr_col_min <= col < curr_col_min + edge:
                        score = (128 ** 2) // (edge ** 2)  # Compute score
                        
                        # Check if the patch already exists in the dictionary
                        patch_key = (curr_row_min, curr_row_min + edge, curr_col_min, curr_col_min + edge)
                        
                        if patch_key in sorted_patches:
                            # If it exists, add the score to the existing value
                            sorted_patches[patch_key] = (
                                sorted_patches[patch_key][0] + score,  # Add the score
                                sorted_patches[patch_key][1] + 
                                dst[row, col]  # Add the corresponding pixel values
                            )
                        else:
                            # If it doesn't exist, initialize with the score and pixel value
                            sorted_patches[patch_key] = (
                                score + 1, dst[row, col]
                            )
                        
                curr_col_min += edge // 2  # Move sliding window horizontally
            curr_col_min = col_min  # Reset column for next row
            curr_row_min += edge // 2  # Move sliding window vertically


        edge += 16  # Update edge to avoid infinite loop

    # Sort in descending order based on score
    sorted_patches = sorted(sorted_patches.items(), key=lambda x: x[1][0], reverse=True)

    # Take the top 10 patches
    top_10 = sorted_patches[:10]
    #print("length of sorted",len(top_10))
    # Sort top 10 based on `dst` value
    top_10_sorted_by_dst = sorted(top_10, key=lambda x: x[1][1], reverse=True)

    # Take the top 5 patches
    top_5 = top_10_sorted_by_dst[:5]
    #print("patch in extractfunrt",top_5[0][0])
    # Return final top 5
    return [patch[0] for patch in top_5]  # Return only the patch coordinates

import numpy as np

def extract_cropped2(matches, dst):
    """
    Extracts patches from the image based on matches, computes scores, and returns top patches.
    
    Args:
    - matches: List of keypoints [(row, col), ...]
    - dst: Destination image (used for calculating pixel values)

    Returns:
    - Top 1 patch with its coordinates
    """
    sorted_patches = {}
    edge = 240

    # Get cropping dimensions
    row_min, row_max, col_min, col_max = crop_dimensions(matches)
    print(f"Crop bounds: row_min={row_min}, row_max={row_max}, col_min={col_min}, col_max={col_max}")
    print(f"Keypoints given: {matches}")

    # Case 1: If crop area is smaller than 128x128, find a 128x128 region that maximizes keypoints coverage
    if (row_max - row_min < edge) or (col_max - col_min < edge):
        # Compute centroid of keypoints
        rows, cols = zip(*matches)
        centroid_row, centroid_col = np.mean(rows), np.mean(cols)

        # Define 128x128 patch centered around the centroid
        new_row_min = max(0, int(centroid_row - edge // 2))
        new_col_min = max(0, int(centroid_col - edge // 2))
        new_row_max = new_row_min + edge
        new_col_max = new_col_min + edge

        # Ensure the patch is within image bounds
        img_height, img_width = dst.shape[:2]
        if new_row_max > img_height:
            new_row_min = img_height - edge
            new_row_max = img_height
        if new_col_max > img_width:
            new_col_min = img_width - edge
            new_col_max = img_width

        print(f"Small area detected, selecting a 128x128 patch centered at ({centroid_row}, {centroid_col})")
        return (new_row_min, new_row_max, new_col_min, new_col_max)

    # Case 2: Normal sliding window search
    curr_row_min, curr_col_min = row_min, col_min
    while curr_row_min <= row_max - edge:
        while curr_col_min <= col_max - edge:
            score = sum(1 for row, col in matches if curr_row_min <= row <= curr_row_min + edge and curr_col_min <= col <= curr_col_min + edge)
            patch_key = (curr_row_min, curr_row_min + edge, curr_col_min, curr_col_min + edge)
            if score > 0:
                sorted_patches[patch_key] = score
            curr_col_min += edge // 2  
        curr_col_min = col_min  
        curr_row_min += edge // 2  

    sorted_patches = sorted(sorted_patches.items(), key=lambda x: x[1], reverse=True)
    #print("Length of sorted patches:", len(sorted_patches))

    return sorted_patches[0][0] if sorted_patches else None

def get_bbox(matches, dimensions, kp1, kp2,dst2):
    # Unpack dimensions
    row_min, row_max, col_min, col_max = dimensions

    img1_features = []
    img2_features = []

    for match in matches:
        query_idx = match.queryIdx  # Index of keypoint in image 1
        train_idx = match.trainIdx  # Index of keypoint in image 2

        # Get keypoint location from image 1
        col,row = kp1[query_idx].pt  # (x, y) coordinate

        # Check if it's within the bounding box
        if row_min <= row <= row_max and col_min <= col <= col_max:
            img1_features.append((row,col))
            col2,row2 = kp2[train_idx].pt
            # Get the corresponding keypoint location in image 2
            img2_features.append((row2,col2)) 
    #print("second vector",img2_features)
    # Compute bounding box in image 2 using the matched points
    if img2_features:
        bbox_image2 = crop_dimensions(img2_features)
    else:
        bbox_image2 = None  # No matches found in bbox
    patch_dimensions=extract_cropped2(img2_features,dst2)
    print(bbox_image2)
    return patch_dimensions

def construct_matrix(src_list, dst_list):
    """
    Construct a matrix for calculating the perspective transformation.
    src_list and dst_list are lists of tuples in (col, row) (x, y) format.
    """
    # Initialize an empty matrix
    A = np.zeros((0, 9))
    
    # Iterate through the source and destination points
    for (xs, ys), (xd, yd) in zip(src_list, dst_list):
        # Construct the rows for the matrix
        first_row = [xs, ys, 1, 0, 0, 0, -xd * xs, -xd * ys, -xd]
        second_row = [0, 0, 0, xs, ys, 1, -yd * xs, -yd * ys, -yd]
        
        # Append the rows to the matrix
        A = np.vstack([A, first_row, second_row])
    
    # Compute A_T * A
    
    ATA = np.dot(A.T, A)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(ATA)
    
    # Find the eigenvector corresponding to the smallest eigenvalue
    min_eigenvalue_index = np.argmin(eigenvalues)
    h = eigenvectors[:, min_eigenvalue_index]
    H = h.reshape((3, 3))
    return  H


def extend_canvas(img1, img2, h):
    # Get dimensions of img1 and img2
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Compute the warped corners of img1
    corners = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]], dtype=np.float32)
    warped_corners = cv2.perspectiveTransform(corners[None, :, :], h)[0]
    #print('shape of warmped',warped_corners.shape)
    tmp_corners=corners
    #print('shape of tmpwarmped',tmp_corners.shape)
    #tmp_corners=find_corners(tmp_corners,h)
    #print('new corners are ',warped_corners)
    # Find the bounding box of the warped corners
    x_min = min(warped_corners[:, 0].min(), 0)
    y_min = min(warped_corners[:, 1].min(), 0)
    x_max = max(warped_corners[:, 0].max(), w2)
    y_max = max(warped_corners[:, 1].max(), h2)

    # Compute the size of the new canvas
    new_width = int(x_max - x_min)
    new_height = int(y_max - y_min)

    # Compute the translation matrix to shift the content to fit in the canvas
    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    #print("xmin and ymin ",x_min , y_min)
    # Warp img1 onto the extended canvas
    extended_h = translation @ h  # Combine translation with the homography
    #warped_img1 = cv2.warpPerspective(img1,extended_h, (new_width, new_height))
    warped_img1 = cv2.warpPerspective(img1,extended_h, (new_width, new_height))

    # Place img2 onto the extended canvas
    canvas = warped_img1.copy()
    canvas[-int(y_min): -int(y_min)+h2, -int(x_min):-int(x_min) + w2] = img2
    #canvas[0:h2, 0:w2] = img2

    return canvas


def common_patches(img_1_path,img_2_path,img_dir):
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

    """
    Read a set of images for Panorama stitching
    
    #root = "/home/speedracer1702/Projects/academic/Computer_Vision/"
    img_dir = "../Data/Set1/"

    img_1_path = img_dir + "1.jpg"
    img_2_path = img_dir + "2.jpg"
    
    """
    print('img1 path',img_1_path)
    print('img2 path',img_2_path)
    img1 = cv2.imread(img_1_path)
    img2 = cv2.imread(img_2_path)

    """
	Corner Detection
	Save Corner detection output as corners.png
	"""

    dst1 = get_harris_corners(img1, show=False)
    dst2 = get_harris_corners(img2, show=False)

    """
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""

    ANMS_keypoints_1 = ANMS(img1,dst1, 250)
    ANMS_keypoints_2 = ANMS(img2,dst2, 250)

    # cv2.destroyAllWindows()
    

    """
	Feature Descriptors
	Save Feature Descriptor output as FD.png
	"""

    feature_descriptors_1 = get_feature_descriptors(img1, ANMS_keypoints_1)
    feature_descriptors_2 = get_feature_descriptors(img2, ANMS_keypoints_2)


    """
	Feature Matching
	Save Feature Matching output as matching.png
	"""

    matches = match_features(feature_descriptors_1, feature_descriptors_2)
    kp1 = [cv2.KeyPoint(x=col, y=row, size=10) for row, col in ANMS_keypoints_1]
    kp2 = [cv2.KeyPoint(x=col, y=row, size=10) for row, col in ANMS_keypoints_2]
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(10,5))
    plt.imshow(img3)
    plt.title("matches")
    plt.legend()
    #plt.savefig("./YourDirectoryID_p1/Phase1/Data/Train/mid_outputs/anms_image.jpg")
    plt.show()
    
    ## GET THE MATCHES FOR EXTRACT FUNCTION ##
    check_repeat = {}
    
    for m in matches:
        key = tuple(ANMS_keypoints_2[m.trainIdx])  # Corrected to use ANMS_Keypoints_2
        if key not in check_repeat:
            check_repeat[key] = [1, ANMS_keypoints_1[m.queryIdx]]  # Store count and keypoint from image 1
        else:
            check_repeat[key][0] += 1  # Increment count
    
    patch_matches = [val[1] for val in check_repeat.values() if val[0] == 1]
    patch_dimensions=extract_croppedimage(patch_matches, dst1)
    # take best one now 
    
    patch_dimensions1=patch_dimensions[0]
    p1row_min,p1row_max,p1col_min,p1col_max=patch_dimensions1
    img1_corners=[
        (p1row_min, p1col_min),  # Corner A
        (p1row_min, p1col_max),  # Corner B
        (p1row_max, p1col_min),  # Corner C
        (p1row_max, p1col_max),  # Corner D
    ]
    patchB=img1[p1row_min:p1row_max,p1col_min:p1col_max]
    
    patch_dimensions2=get_bbox(matches, patch_dimensions1, kp1, kp2,dst2)
    p2row_min,p2row_max,p2col_min,p2col_max=patch_dimensions2
    img2_corners=[
        (p2row_min, p2col_min),  # Corner A
        (p2row_min, p2col_max),  # Corner B
        (p2row_max, p2col_min),  # Corner C
        (p2row_max, p2col_max),  # Corner D
    ]
    p2row_min, p2row_max, p2col_min, p2col_max = map(int, patch_dimensions2)
    patchA=img2[p2row_min:p2row_max,p2col_min:p2col_max]
    #steps further
    #translate first
    #the dimensions are outputs of it, now see , we put this patch in image 1
    #put these patches in test folder and get the Hpt and save it in stitchoutput...and bring them 
    #now you know the corners in patchA which is in image2, which got ,use Hpt to get other corners , use getperspective to get
    #warping for going from corners of patch A to new corners..that is same as H from image1 to image2
    #put it in extendcanvass..
    
        
   
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.imshow(img3)
    # plt.show()
    
    

        # Create a figure with 1 row and 2 columns for side-by-side plots
    plt.figure(figsize=(10, 5))
    
    # Display patchB in the first subplot
    plt.subplot(1, 2, 1)  # (rows, columns, index)
    plt.imshow(patchB)
    plt.title('patchB')
    
    # Display patchA in the second subplot
    plt.subplot(1, 2, 2)
    plt.imshow(patchA)
    plt.title('patchA')
    
    # Show the plots side by side
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.legend()
    plt.show()
        
    # Define the save directory
    save_dir = img_dir+"/Patches"
    os.makedirs(save_dir, exist_ok=True)  # Create dir if it doesn't exist
    
    # Resize patches to 128x128
    patchA_resized = cv2.resize(patchA, (128, 128))
    patchB_resized = cv2.resize(patchB, (128, 128))
    
    # Save the resized images
    cv2.imwrite(os.path.join(save_dir, "patchA.jpg"), patchA_resized)
    cv2.imwrite(os.path.join(save_dir, "patchB.jpg"), patchB_resized)
    
    #print("Patches saved in", save_dir)

    #print("dimensions of patch",patch_dimensions1)
    return img1_corners,img2_corners



def get_Hfinal(Hpath,img1_corners,img2_corners):
    """
    Reads H_values from Hpath, computes cornersA and cornersB, 
    and swaps rows and columns for homography calculation.

    Args:
    - Hpath: Path to the H4pt_labels.txt file.
    - patch_dimensions: List of lists [[row_min, row_max, col_min, col_max], ...]

    Returns:
    - h: Homography matrix computed from corresponding corners.
    """
    with open(Hpath, "r") as file:
        H_values = list(map(int, map(float, file.readline().split())))

    # Assuming patch_dimensions contains only one patch
    
    ###calculation Hr1---- src -- img1corners and dst -- 
    dl_cords=[(0,0),(0,128),(128,0),(128,128)]
    Hr1=cv2.getPerspectiveTransform(np.float32(rowcol_to_colrow(img1_corners)), np.float32(rowcol_to_colrow(dl_cords)))
    Hr2_inv=cv2.getPerspectiveTransform(np.float32(rowcol_to_colrow(dl_cords)), np.float32(rowcol_to_colrow(img2_corners)))
    
    #Put PatchB from im
    # Define cornersA (original patch corners) as tuples
    [
        (row_min, col_min),  # Corner A
        (row_min, col_max),  # Corner B
        (row_max, col_min),  # Corner C
        (row_max, col_max),  # Corner D
    ]=dl_cords

    #Define cornersB (adjusted with H_values) as tuples
    cornersB = [
        (row_min + H_values[0], col_min + H_values[1]),  # Corner A
        (row_min + H_values[2], col_max + H_values[3]),  # Corner B
        (row_max + H_values[4], col_min + H_values[5]),  # Corner C
        (row_max + H_values[6], col_max + H_values[7]),  # Corner D
    ]
    # cornersB = [
    #     (row_min + H_values[1], col_min + H_values[0]),  # Corner A
    #     (row_min + H_values[3], col_max + H_values[2]),  # Corner B
    #     (row_max + H_values[5], col_min + H_values[4]),  # Corner C
    #     (row_max + H_values[7], col_max + H_values[6]),  # Corner D
    # ]

    # Swap row and column for both sets
    src = rowcol_to_colrow(dl_cords)
    dst = rowcol_to_colrow(cornersB)

    # Compute homography using OpenCV
    Hw = cv2.getPerspectiveTransform(np.float32(dst), np.float32(src))
    #h=construct_matrix(src,dst)
    #h=np.linalg.pinv(h)
    h=Hr2_inv@Hw@Hr1
    #h=Hw
    #print("H value",h)
    return h  # Return only the homography matrix for the given patch

def stitch_operation(img_1_path, img_2_path, h, stitched_img_path):
    img1 = cv2.imread(img_1_path)
    img2 = cv2.imread(img_2_path)

    if img1 is None or img2 is None:
        print("Error: One of the images could not be loaded.")
        return

    combined_image = extend_canvas(img1, img2, h)

    # Ensure the provided directory exists
    os.makedirs(os.path.dirname(stitched_img_path), exist_ok=True)

    # Save stitched image at the specified path
    cv2.imwrite(stitched_img_path, combined_image)

    # Display the stitched image
    plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display
    plt.axis("off")  # Hide axes for cleaner visualization
    plt.show()

# def automate_stitching(train_path):
#     # Direct path to a specific set (e.g., Set1)
#     set_path = train_path  # You can specify a folder like Set1, etc.

#     # Get all image filenames in the set, sorted numerically
#     excluded_files = ["stitched_result.jpg", "stitched_image.jpg"]  # Add other filenames to exclude as needed
    
#     # Get all image filenames in the folder, sorted numerically, but exclude specific files
#     image_files = sorted(
#         [f for f in os.listdir(train_path) if f.endswith('.jpg') and f not in excluded_files],
#         key=lambda x: int(x.split('.')[0])
#     )
#     # Initialize the first image path
#     current_image_path = os.path.join(set_path, image_files[0])
    
#     # Stitch all images in the set one by one
#     for i in range(1, len(image_files)):
#         next_image_path = os.path.join(set_path, image_files[i])
        
#         # Call your stitch_operation function
#         #print(f"Stitching {current_image_path} with {next_image_path}")
#         Hpath= "../Data/Test/Labels/H4pt_labels.txt"
#         # img1_corners,img2_corners=common_patches(img_1_path,img_2_path,img_dir)
#         #print("selected_points",patch_dim)
#         # h=get_Hfinal(Hpath,img1_corners,img2_corners)
#         # stitch_operation(img_1_path,img_2_path,h)
#         current_image_path = stitch_operation(current_image_path, next_image_path, set_path)
    
#     # Save the final stitched image
#     final_image_path = os.path.join(set_path, "stitched_result.jpg")
#     #print(f"Saving the final stitched image at: {final_image_path}")
#     #cv2.imwrite(final_image_path, current_image_path)
# # '''
# def main():
#     # Add any Command Line arguments here
#     # Parser = argparse.ArgumentParser()
#     # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

#     # Args = Parser.parse_args()
#     # NumFeatures = Args.NumFeatures

#     """
#     Read a set of images for Panorama stitching
#     """
#     #root = "/home/speedracer1702/Projects/academic/Computer_Vision/"
   
#     train_path = "../Data/Set2"
#     img_1_path = train_path + "/stitched_result.jpg"
#     img_2_path = train_path + "/3.jpg"
#     img_dir=train_path
#     Hpath= "../Data/Test/Labels/H4pt_labels.txt"
#     img1_corners,img2_corners=common_patches(img_1_path,img_2_path,img_dir)
#     #print("selected_points",patch_dim)
#     h=get_Hfinal(Hpath,img1_corners,img2_corners)
#     stitch_operation(img_1_path,img_2_path,h)
    
#     #automate_stitching(train_path)


# if __name__ == "__main__":
#     main()

    