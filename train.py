import os
import argparse
import torch
import cv2
import glob
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.losses.basic_loss import L1Loss
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.data.transforms import augment, paired_random_crop

# Custom implementation of bgr2ycbcr
def bgr2ycbcr(img, y_only=False):
    """Convert a BGR image to YCbCr.
    Args:
        img (ndarray): Input image with shape (H, W, C) in BGR order.
        y_only (bool): Whether to return only the Y channel.
    Returns:
        ndarray: Output image in YCbCr format.
    """
    img = img.astype(np.float32) / 255.0
    if y_only:
        y = 16.0 + 65.481 * img[:, :, 2] + 128.553 * img[:, :, 1] + 24.966 * img[:, :, 0]
        y = np.clip(y, 16.0, 235.0)
        return y[..., None]
    else:
        y = 16.0 + 65.481 * img[:, :, 2] + 128.553 * img[:, :, 1] + 24.966 * img[:, :, 0]
        cb = 128.0 - 37.797 * img[:, :, 2] - 74.203 * img[:, :, 1] + 112.0 * img[:, :, 0]
        cr = 128.0 + 112.0 * img[:, :, 2] - 93.786 * img[:, :, 1] - 18.214 * img[:, :, 0]
        ycbcr = np.stack([y, cb, cr], axis=-1)
        ycbcr = np.clip(ycbcr, 16.0, 240.0)
        return ycbcr

def paired_paths_from_folder(lq_folder, gt_folder, lq_ext='.jpeg', gt_ext='.jpeg'):
    """Generate paired paths from folders with different extensions."""
    # Print the paths being searched
    print(f"Searching for LR images in: {lq_folder}")
    print(f"Searching for HR images in: {gt_folder}")
    
    lq_paths = sorted(glob.glob(os.path.join(lq_folder, f'*{lq_ext}')))
    print(f"Found {len(lq_paths)} LR images in {lq_folder}")
    
    pairs = []
    for lq_path in lq_paths:
        base = os.path.splitext(os.path.basename(lq_path))[0]
        gt_path = os.path.join(gt_folder, f"{base}{gt_ext}")
        if os.path.exists(gt_path):
            pairs.append({'lq_path': lq_path, 'gt_path': gt_path})
        else:
            print(f"Warning: No matching HR image found for {lq_path}")
    
    print(f"Found {len(pairs)} valid LR-HR pairs")
    return pairs

class CustomPairedDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = {'type': 'disk'}
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)
        
        self.paths = paired_paths_from_folder(
            opt['dataroot_lq'],
            opt['dataroot_gt'],
            lq_ext='.jpeg',  # Update to .jpeg
            gt_ext='.jpeg'   # Update to .jpeg
        )
        
        if len(self.paths) == 0:
            raise ValueError("No valid LR-HR image pairs found. Check your dataset paths and file extensions.")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # Initialize file client
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        
        # Load GT image
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        
        # Load LQ image
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # Ensure dimensions are divisible by scale
        h, w = img_gt.shape[:2]
        new_h = h - h % scale
        new_w = w - w % scale
        img_gt = cv2.resize(img_gt, (new_w, new_h))
        
        # Resize LQ to match correct scale ratio
        lq_h, lq_w = new_h // scale, new_w // scale
        img_lq = cv2.resize(img_lq, (lq_w, lq_h), interpolation=cv2.INTER_CUBIC)

        # Data augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # Color space conversion to Y-channel if needed
        if self.opt.get('color', 'rgb') == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)
            img_lq = bgr2ycbcr(img_lq, y_only=True)

        # Convert to tensor and normalize
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        
        if self.mean is not None or self.std is not None:
            # Normalize if mean/std provided
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

def train(args):
    # Initialize model
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=args.scale
    )
    
    # Configure dataset
    dataset_opt = {
        'dataroot_gt': args.hr_dir,
        'dataroot_lq': args.lr_dir,
        'phase': 'train',
        'gt_size': 128,
        'scale': args.scale,
        'use_hflip': True,
        'use_rot': True,
        'color': 'rgb'
    }
    
    # Create data loader
    train_dataset = CustomPairedDataset(dataset_opt)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Loss and optimizer
    criterion = L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            lq = batch['lq']
            gt = batch['gt']
            
            # Forward pass
            outputs = model(lq)
            loss = criterion(outputs, gt)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update progress
            total_loss += loss.item()
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{(total_loss/(progress_bar.n+1)):.4f}'
            })
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.checkpoints_dir, f'epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Saved checkpoint to {checkpoint_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Super-Resolution Model')
    parser.add_argument('--lr_dir', type=str, required=True, help='Path to low-resolution images')
    parser.add_argument('--hr_dir', type=str, required=True, help='Path to high-resolution images')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=2, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--scale', type=int, default=4, help='Super-resolution scale factor')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    
    # Start training
    train(args)