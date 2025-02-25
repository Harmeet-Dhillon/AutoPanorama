#!/usr/bin/env python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

import argparse
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.onnx
import torch.optim as optim
from torch.optim import  lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Misc.DataUtils import *
from Misc.MiscUtils import *
from Network.Network import LossFn, SupervisedNet
torch.cuda.empty_cache()

def GenerateBatch(DirNamesTrainA,DirNamesTrainB, CoordinatesDict, MiniBatchSize):
    """
    Inputs:
    BasePath - Path to COCO folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainCoordinates - Coordinatess corresponding to Train
    NOTE that TrainCoordinates can be replaced by Val/TestCoordinatess for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    PatchesBatch - Batch of images
    CoordinatesBatch - Batch of coordinates
    """
    PatchesBatch = []
    CoordinatesBatch = []
    mean = [0.485, 0.456, 0.406] # for standardization  
    std = [0.229, 0.224, 0.225]
    ImageNum = 0
    #print("cordinates length",len(CoordinatesDict))
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(1, len(DirNamesTrainA))
        
        RandImageNameA = DirNamesTrainA[RandIdx-1] + ".jpg"
        RandImageNameB = DirNamesTrainB[RandIdx-1] + ".jpg"

        ImageNum += 1

        ##########################################################
        # Add any standardization or data augmentation here!
        ##########################################################
        I1 = np.float32(cv2.imread(RandImageNameA))
        I2 = np.float32(cv2.imread(RandImageNameB))
    
        #normalizing two images ...but be prepared as normalization may reduce accuracy ..may have effect on features laters
        # I1 = (I1 - mean) / std
        # I2 = (I2 - mean) / std
        Coordinates = CoordinatesDict[str(RandIdx)]
        #print("cordinates ",Coordinates)
        If=np.concatenate((I1,I2),axis=-1)
        If=np.transpose(If, (2, 0, 1))
        #print("Tensor shape",If.shape)
        # Append All Images and Mask
        PatchesBatch.append(torch.from_numpy(If).float())
        CoordinatesBatch.append(torch.tensor(Coordinates))

    return torch.stack(PatchesBatch), torch.stack(CoordinatesBatch)


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print("Number of Epochs Training will run for " + str(NumEpochs))
    print("Factor of reduction in training data is " + str(DivTrain))
    print("Mini Batch Size " + str(MiniBatchSize))
    print("Number of Training Images " + str(NumTrainSamples))
    if LatestFile is not None:
        print("Loading latest checkpoint with the name " + LatestFile)

def printGPUStats():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    free_mem = torch.cuda.mem_get_info()[0] / 1024**2  # Convert bytes to MB
    print(f"Free GPU memory: {free_mem:.2f} MB")


def TrainOperation(
    NumEpochs,
    MiniBatchSize,
    SaveCheckPoint,
    CheckPointPath,
    DivTrain,
    LatestFile,
    BasePath,
    LogsPath,
    ModelType,
):
    # Predict output with forward pass

    # Obtain training logistics
    (
        DirNamesTrainA,
        DirNamesTrainB,
        DirNamesValA,
        DirNamesValB,
        NumTrainSamples,
        NumValSamples,
        TrainCoordinates,
        ValCoordinates
    ) = SetupAll(BasePath)

    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    if ModelType == "Supervised":
        model = SupervisedNet()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    model.to(device)

    #Optimizer = optim.AdamW(model.parameters(), lr=0.0001,weight_decay=0.01)
    Optimizer = optim.SGD(model.parameters(),lr=0.005, momentum=0.9)
    #scheduler = lr_scheduler.ExponentialLR(Optimizer, gamma=0.9)
    # Tensorboard
    Writer = SummaryWriter(LogsPath)

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + ".ckpt")
        # Extract only numbers from the name
        StartEpoch = int("".join(c for c in LatestFile.split("a")[0] if c.isdigit()))
        model.load_state_dict(CheckPoint["model_state_dict"])
        print("Loaded latest checkpoint with the name " + LatestFile + "....")
    else:
        StartEpoch = 0
        print("New model initialized....")

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(StartEpoch, NumEpochs), desc="Epochs"):
        
        print("Epoch number: ", epoch)

        # Training
        
        model.train()

        train_loss = 0.0
        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)

        for _ in tqdm(range(NumIterationsPerEpoch), desc="Training Iterations"):
            
            StackedTrainPatches, TrainCoordinatesBatch = GenerateBatch(DirNamesTrainA,
                                                             DirNamesTrainB,
                                                             TrainCoordinates,
                                                             MiniBatchSize)
            # Predict output with forward pass
            StackedTrainPatches = StackedTrainPatches.to(device)
            TrainCoordinatesBatch = TrainCoordinatesBatch.to(device)
            PredicatedCoordinatesBatch = model(StackedTrainPatches)
            LossThisBatch = LossFn(PredicatedCoordinatesBatch, TrainCoordinatesBatch)

            # Update weights
            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.1)


            train_loss += LossThisBatch.detach().cpu().numpy()

            # delete training data from gpu
            # del StackedTrainPatches, TrainCoordinatesBatch
    
        # Free GPU memory for validation
        # torch.cuda.empty_cache()

        average_epoch_training_loss = train_loss/NumIterationsPerEpoch
        Writer.add_scalar('Train_loss', average_epoch_training_loss, epoch)
        train_losses.append(average_epoch_training_loss)

        # Validation

        model.eval()

        valIterations = int(NumValSamples/MiniBatchSize)

        val_loss_outputs = []

        for _ in tqdm(range(valIterations), desc = "Val Iterations"):

            StackedValPatches, ValCoordinatesBatch = GenerateBatch(DirNamesValA,
                                                                    DirNamesValB,
                                                                    ValCoordinates,
                                                                    MiniBatchSize)
            StackedValPatches = StackedValPatches.to(device)
            ValCoordinatesBatch = ValCoordinatesBatch.to(device)

            val_loss_outputs.append(model.validation_step(StackedValPatches,
                                                          ValCoordinatesBatch))
            # delete data from gpu
            # del StackedValPatches, ValCoordinatesBatch
            # torch.cuda.empty_cache()

        avg_validation_loss = model.validation_epoch_end(val_loss_outputs)

        # Tensorboard
        val_losses.append(avg_validation_loss)

        #####changed here to plot it 
        results={
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        PlotsPath = "../Data/Plots"
        os.makedirs(PlotsPath, exist_ok=True)

        plotepochResults(PlotsPath,ModelType,results,epoch)

        Writer.add_scalar("Val_loss", avg_validation_loss, epoch)

        # If you don't flush the tensorboard doesn't update until a lot of iterations!
        Writer.flush()

        # Save model every 100 epoch

        if epoch % SaveCheckPoint == 0 or epoch == NumEpochs-1:  # Save model only if epoch is a multiple of 100
            SaveName = CheckPointPath + str(epoch) + "model.ckpt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": Optimizer.state_dict(),
                    "loss": LossThisBatch,
                },
                SaveName,
            )
            print("\n" + SaveName + " Model Saved...")
        #scheduler.step()
        printGPUStats()

    # Save model as onnx for visualization
    dummy_input = torch.randn(1, 6, 128, 128).to(device)
    onnx_path = CheckPointPath + ModelType + ".onnx"
    model.eval()
    torch.onnx.export(model, dummy_input, onnx_path, export_params=True, opset_version=11)
    

    return results

def plotepochResults(PlotsPath, ModelName, results,epoch):
    # Extract results
    train_losses = results['train_losses']
    test_losses = results['val_losses']

    NumEpochs = epoch+1

    # Plot losses
    plt.plot(range(NumEpochs), train_losses, label='Train Loss')
    plt.plot(range(NumEpochs), test_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Over Epochs')

    # Save the plot
    loss_plot_path = f"{PlotsPath}/{ModelName}_TrainTest_epochloss.png"
    plt.savefig(loss_plot_path)
    

def plotResults(PlotsPath, ModelName, results):
    # Extract results
    train_losses = results['train_losses']
    test_losses = results['val_losses']

    NumEpochs = len(results['train_losses'])

    # Plot losses
    plt.plot(range(NumEpochs), train_losses, label='Train Loss')
    plt.plot(range(NumEpochs), test_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Over Epochs')

    # Save the plot
    loss_plot_path = f"{PlotsPath}/{ModelName}_TrainTest_loss.png"
    plt.savefig(loss_plot_path)
    plt.show()
   
    


def main():
    """
    Inputs:
    # None
    # Outputs:
    # Runs the Training and testing code based on the Flag
    #"""
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--BasePath",
        default="../Data",
        help="Base path of images, Default:../Data/Train",
    )
    Parser.add_argument(
        "--CheckPointPath",
        default="../Data/Checkpoints/",
        help="Path to save Checkpoints, Default: ../Data/Checkpoints/",
    )
    Parser.add_argument(
        "--ModelType",
        default="Supervised",
        help="Model type, Supervised or Unsupervised? Choose from Supervised and Unsupervised, Default:Supervised",
    )
    Parser.add_argument(
        "--NumEpochs",
        type=int,

        default=100,

        help="Number of Epochs to Train for, Default:50",
    )
    Parser.add_argument(
        "--DivTrain",
        type=int,
        default=1,
        help="Factor to reduce Train data by per epoch, Default:1",
    )
    Parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=64,

        help="Size of the MiniBatch to use, Default:1",
    )
    Parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=0,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    )
    Parser.add_argument(
        "--LogsPath",
        default="../Data/Logs/",
        help="Path to save Logs for Tensorboard, Default=../Data/Logs/",
    )
    print("the path")
    Parser.add_argument(
        "--SaveCheckpoints",
        default="../Data/Logs/",
        help="Path to save Logs for Tensorboard, Default=../Data/Logs/",
    )

    print("print ho raha hai",os.getcwd())

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType

    # If CheckPointPath doesn't exist make the path
    if not (os.path.isdir(CheckPointPath)):
        os.makedirs(CheckPointPath)

    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    # Save checkpoint every SaveCheckPoint epochs
    SaveCheckPoint = 50

    results = TrainOperation(
        NumEpochs,
        MiniBatchSize,
        SaveCheckPoint,
        CheckPointPath,
        DivTrain,
        LatestFile,
        BasePath,
        LogsPath,
        ModelType,
    )

    # Plot results using matplotlib

    PlotsPath = "../Data/Plots"
    os.makedirs(PlotsPath, exist_ok=True)

    plotResults(PlotsPath, ModelType, results)


if __name__ == "__main__":
    main()
