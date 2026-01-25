import os
import argparse
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys

#adding the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from landslide_dataloader import LandslideDataset
from landslide_model import UNet
from landslide_trainer import train_one_epoch, validate_one_epoch
from utils.metrics import DiceLoss

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device in use: {device}")

    #training image and mask paths
    train_img_path = os.path.join(args.data_path, "TrainData", "img")
    train_mask_path = os.path.join(args.data_path, "TrainData", "mask")
    
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    #prepare data loaders using filenames from utils/filenames/
    train_filenames = os.path.join(args.filenames_dir, "train.txt")
    val_filenames = os.path.join(args.filenames_dir, "val.txt")

    train_dataset = LandslideDataset(train_img_path, train_mask_path, train_filenames, augment=True)
    valid_dataset = LandslideDataset(train_img_path, train_mask_path, val_filenames, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    #initialize model, loss, and optimizer
    model = UNet(in_channels=14, out_channels=1).to(device)
    
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"loading checkpoint from {args.checkpoint_path}")
        model.load_state_dict(torch.load(args.checkpoint_path))

    loss_fn = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    #main training loop
    for epoch in range(args.epochs):
        train_loss, train_prec, train_rec, train_f1 = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device)
        val_loss, val_prec, val_rec, val_f1 = validate_one_epoch(
            model, valid_loader, loss_fn, device)
        
        print(f"epoch {epoch+1}/{args.epochs} - train loss: {train_loss:.4f}, val loss: {val_loss:.4f}")

    #save trained model
    torch.save(model.state_dict(), args.model_path)
    print(f"model saved to {args.model_path}")

def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_img_path = os.path.join(args.data_path, "TestData", "img")
    os.makedirs(args.output_path, exist_ok=True)

    #load model
    model = UNet(in_channels=14, out_channels=1).to(device)
    if not os.path.exists(args.model_path):
        print(f"error: model file {args.model_path} not found.")
        return
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    #run predictions on test data
    test_files = sorted([f for f in os.listdir(test_img_path) if f.endswith(".h5")])
    print(f"running predictions on {len(test_files)} test files...")

    with torch.no_grad():
        for f_name in test_files:
            path = os.path.join(test_img_path, f_name)
            with h5py.File(path, "r") as f:
                image = f["img"][:]
            
            #normalize
            min_val, max_val = image.min(), image.max()
            if max_val > min_val:
                image = (image - min_val) / (max_val - min_val)
            else:
                image = np.zeros_like(image)
            
            input_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0).to(device)
            output = model(input_tensor).cpu().squeeze().numpy()
            
            #save prediction as h5
            save_path = os.path.join(args.output_path, f_name)
            with h5py.File(save_path, "w") as f:
                f.create_dataset("mask", data=output)

    print(f"predictions completed. saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Landslide Prediction Training/Testing Script")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="mode: train or test")
    parser.add_argument("--data_path", type=str, required=True, help="path to the dataset root")
    parser.add_argument("--filenames_dir", type=str, default="utils/filenames", help="directory containing train.txt and val.txt")
    parser.add_argument("--model_path", type=str, default="models/unet_landslide.pth", help="path to save/load the model")
    parser.add_argument("--output_path", type=str, default="predictions", help="path to save predictions (test mode)")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="path to a checkpoint to resume training")
    parser.add_argument("--epochs", type=int, default=50, help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    else:
        test(args)
