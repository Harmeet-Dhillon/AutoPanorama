"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

import os
import random
import sys

import cv2
import numpy as np
import PIL
import skimage

# Don't generate pyc codes
sys.dont_write_bytecode = True


def SetupAll(BasePath):

    # Train

    TrainDirNamesPatchesA,TrainDirNamesPatchesB = SetupDirNames(BasePath)
    TrainLabelsPath = BasePath + "/Train/TxtFiles/Labels.txt"
    TrainLabels = ReadLabels(TrainLabelsPath)


    # Val
    ValDirNamesPatchesA,ValDirNamesPatchesB = SetupDirNames(BasePath,val=True)
    ValLabelsPath = BasePath + "/Val/TxtFiles/Labels.txt"
    ValLabels = ReadLabels(ValLabelsPath)
    
    # Image Input Shape
    
    NumTrainSamples = len(TrainDirNamesPatchesA)
    NumValSamples = len(ValDirNamesPatchesA)
    print("Lenght of Val",len(ValLabels))
    return (
        TrainDirNamesPatchesA,
        TrainDirNamesPatchesB,
        ValDirNamesPatchesA,
        ValDirNamesPatchesB,
        NumTrainSamples,
        NumValSamples,
        TrainLabels,
        ValLabels
    )

def ReadLabels(LabelsPath):
    if not os.path.isfile(LabelsPath):
        print("ERROR: Train Labels do not exist in " + LabelsPath)
        sys.exit()
    else:
        # Open the file and read all lines
        with open(LabelsPath, "r") as file:
            Labels = {}
            for line in file:
                # Split the line by spaces
                parts = line.split()
                
                # The first part is the image name (e.g., "image1")
                image_name = parts[0]
                
                # The rest are the 8 float values, which will be stored as a list
                coordinates = list(map(float, parts[1:]))
                
                # Save the image name and corresponding coordinates as a dictionary
                Labels[image_name] = coordinates
    # give back dictionary 
    return Labels


def SetupDirNames(BasePath,val=False):
    
    """
    Inputs:
    BasePath is the base path where Images are saved without "/" at the end
    Outputs:
    Writes a file ./TxtFiles/DirNames.txt with full path to all image files without extension
    """
    if not val:
        DirNamesPatchesA = ReadDirNames(BasePath + "/Train/TxtFiles/DirNamesA.txt")
        DirNamesPatchesB = ReadDirNames(BasePath + "/Train/TxtFiles/DirNamesB.txt")
    else:
        DirNamesPatchesA = ReadDirNames(BasePath + "/Val/TxtFiles/DirNamesA.txt")
        DirNamesPatchesB = ReadDirNames(BasePath + "/Val/TxtFiles/DirNamesB.txt")

    return DirNamesPatchesA,DirNamesPatchesB


def ReadDirNames(ReadPath):
    """
    Reads a file containing image paths and ensures each path starts with "./".
    """
    with open(ReadPath, "r") as file:
        DirNames = ["." + line.strip() for line in file.readlines()]  # Add "." to each line

    return DirNames