"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

import sys

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#import kornia  # You can use this to get the transform and warp in this project

# Don't generate pyc codes
sys.dont_write_bytecode = True
loss_fn=nn.MSELoss()
def LossFn(predictedcorners, corners):
    # Calculate the squared differences between predicted and ground truth
    diff = predictedcorners - corners
    
    # Apply L2 norm (Euclidean distance), which is equivalent to the squared difference sum
    # across the batch, followed by taking the square root for the final L2 norm value.
    #loss = torch.norm(diff, p=2, dim=1)  # p=2 for L2 norm, dim=1 for each image
    loss=torch.sqrt(loss_fn(predictedcorners,corners))
    # Calculate mean loss across the mini-batch
    return loss 

class HomographyModel(pl.LightningModule):
    def training_step(self, patchesBatch, cornersBatch):
        predictedCornersBatch = self(patchesBatch)
        loss = float(LossFn(predictedCornersBatch, cornersBatch))
        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, patchesBatch, cornersBatch):
        predictedCornersBatch = self(patchesBatch)
        loss = LossFn(predictedCornersBatch, cornersBatch)
        return loss.detach().cpu().numpy()

    def validation_epoch_end(self, losses):
        avg_loss = np.mean(np.array(losses, dtype=float))
        return avg_loss

class SupervisedNet(HomographyModel):
    def __init__(self):
        super(SupervisedNet, self).__init__()

        dropout_percentage = 0.5

        # CNN Feature Extractor
        layers = []
        in_channels = 6  # 2 channels as we are combining xa and xb (grayscale patches)
        self.arch = [64, 64, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128,'M']
        self.mcount = sum(1 for x in self.arch if x == 'M')
        
        # Build convolutional layers
        for x in self.arch:
            if x != 'M':
                layers.append(nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False))
                layers.append(nn.BatchNorm2d(x))
                layers.append(nn.ReLU(inplace=True))  # Add activation function after conv

                #layers.append(nn.Dropout2d(p=dropout_percentage*2))  # Add dropout after each conv layer            #DROPOUT

                in_channels = x
            else:
                layers.append(nn.MaxPool2d(kernel_size=2))

        self.combine = nn.Sequential(*layers)
        f_size = 128 // 2**self.mcount
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(f_size*f_size*128, 1024)
        self.fc2 = nn.Linear(1024, 8)  # Output 8 values for translation vector (CB - CA)
        self.dropout_fc = nn.Dropout(p=dropout_percentage)  # Add dropout after the fully connected layer            #DROPOUT

    def forward(self, x, unsupervised=False):
        """
        Input:
        xa is a MiniBatch of image A 
        xb is a MiniBatch of image B 
        Outputs:
        out - predicted translation vector (CB - CA)
        """
        if not unsupervised:
            # Concatenate the two grayscale patches (xa and xb) along the channel dimension if taking grayscale images
            #combined = torch.cat((x), dim=1)  # Concatenate along channels (2 channels)

            # Pass the combined input through the CNN layers
            x = self.combine(x)
            x = x.view(x.shape[0], -1) 
            x = self.dropout_fc(x)  # Apply dropout after fc   
            x = self.fc1(x)                                      #DROPOUT
            x= self.fc2(x)  # Output the 8 values representing the translation vector

        return x
    
def normalized_inverse_matrix(H_pred, cornersA):
    """
    inputs:
        H_pred: From tensor DLT
        image_a: patch a
    """

    # Assume image is given as tensor in row,col format

    # Extracting the coordinates of the corners
    y_coords = cornersA[:, 0]  # y coordinates (rows)
    x_coords = cornersA[:, 1]  # x coordinates (columns)

    # Calculating width and height
    width = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)

    M = torch.tensor([[width/2, 0, width/2],
                  [0, height/2, height/2],
                  [0, 0, 0]])
    
    H_inv = torch.linalg.pinv(M)*torch.linalg.pinv(H_pred)*M # does this need to be p_inv or p?

    return H_inv


class UnsupervisedNet(HomographyModel):
    def __init__(self):
        super(UnsupervisedNet, self).__init__()

        dropout_percentage = 0.5

        # CNN Feature Extractor
        layers = []
        in_channels = 6  # 2 channels as we are combining xa and xb (grayscale patches)
        self.arch = [64, 64, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128]
        self.mcount = sum(1 for x in self.arch if x == 'M')
        
        # Build convolutional layers
        for x in self.arch:
            if x != 'M':
                layers.append(nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False))
                layers.append(nn.BatchNorm2d(x))
                layers.append(nn.ReLU(inplace=True))  # Add activation function after conv

                #layers.append(nn.Dropout2d(p=dropout_percentage*2))  # Add dropout after each conv layer            #DROPOUT

                in_channels = x
            else:
                layers.append(nn.MaxPool2d(kernel_size=2))

        self.combine = nn.Sequential(*layers)
        f_size = 128 // 2**self.mcount
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(f_size*f_size*128, 1024)
        self.fc2 = nn.Linear(1024, 8)  # Output 8 values for translation vector (CB - CA)
        self.dropout_fc = nn.Dropout(p=dropout_percentage)  # Add dropout after the fully connected layer            #DROPOUT

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32), nn.ReLU(True), nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    def TensorDLT(H4Pt_pred, A_corners):
        '''
        H4Pt_pred and A_corners need to be on the same device
        '''

        def construct_A_hat_i(u, v, u_prime, v_prime):
            A_hat_i = torch.tensor([
                [0, 0, 0, -u, -v, -1, v_prime * u, v_prime * v],
                [u, v, 1, 0, 0, 0, -u_prime * u, -u_prime * v]
            ], dtype=torch.float32)
            
            return A_hat_i
        
        
        ''' H4Pt_pre and A_corners are assumed to be in row,col format and will be
            converted to x,y format

        '''
        # Reorder form row,col -> x,y

        # Swap columns to convert (y, x) → (x, y)
        H4Pt_pred = H4Pt_pred[:, [1, 0]] # swap columns for all rows
        A_corners = A_corners[:, [1, 0]] # swap columns for all rows

        A_hat = torch.zeros(8, 8, dtype=torch.float32)
        b_hat = torch.zeros(8, 1, dtype=torch.float32)

        B_corners_pred = H4Pt_pred + A_corners

        for i in range(len(A_corners)):  
            A_corner = A_corners[i]  
            B_corner_pred = B_corners_pred[i]  

            # Extract (x, y) from A_corner
            u, v = A_corner[0], A_corner[1]

            # Extract (x', y') from B_corner_pred
            u_prime, v_prime = B_corner_pred[0], B_corner_pred[1]

            A_hat[2*i:2*i+2, :] = construct_A_hat_i(u, v, u_prime, v_prime)
            b_hat[2*i:2*i+2] = torch.tensor([[-v_prime],[u_prime]],
                                            dtype=torch.float32)
            
        H = torch.matmul(torch.linalg.pinv(A_hat), b_hat)
        H = H.reshape(4, 2)

        "Leaving colinear points approach"

        return H
    
    def forward(self, x, cornersA):
        """
        Inputs
            x: Stacked corners of Pa and Pb in row, col format
            cornersA: corners of A in the original image in row, col format
        Outputs
            out: Warped(Pa) and Pa
        """
        # Concatenate the two grayscale patches (xa and xb) along the channel dimension if taking grayscale images
        #combined = torch.cat((x), dim=1)  # Concatenate along channels (2 channels)

        # Pass the combined input through the CNN layers
        x = self.combine(x)
        x = x.view(x.shape[0], -1) 
        x = self.dropout_fc(x)  # Apply dropout after fc   
        x = F.relu(self.fc1(x))                                        #DROPOUT
        H4pt_pred = self.fc2(x)  # Output the 8 values representing the translation vector

        "Spatial transformer network forward function"

        # 0: Localization - Obtain H_pred
        H_pred = self.TensorDLT(H4pt_pred, cornersA)

        # 1: Normalize inverse computation
        normalized_inverse_matrix = (H_pred, cornersA)

        # 2: Generate sampling grid

        # theta is the top 2x3 portion of the normalized homography matrix normalized again by scale
        theta = normalized_inverse_matrix/normalized_inverse_matrix[2, 2]
        theta = normalized_inverse_matrix[:2, :].unsqueeze(0)  # Shape: (1, 2, 3)

        # Theta: Nx2x3 tensor, where N is the batch size
        # size: NxCxHxW tensor
        #       N: Batch size
        #       C: Number of channels
        #       H: height of the input image
        #       W: width of the input image

        # Get image size from the input tensor

        IA = x[:, :3, :, :]  # Extract the first 3 channels for IA (Batch, 3, H, W), since X is a stack of IA and IB
        grid = F.affine_grid(theta, IA.size())

        # 3: Do differenntiable sampling
        warped_image = F.grid_sample(IA, grid)

        return warped_image
    

        
    
        
        


    
    

    
