import os
import argparse
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from tqdm import tqdm

def compute_mae(img1, img2):
    """Compute Mean Absolute Error (MAE) between two images."""
    return np.mean(np.abs(img1 - img2))

def calculate_metrics(gt_file, sr_file, scale, output_file=None):
    # Read ground truth and super-resolved images
    img_gt = cv2.imread(gt_file)
    img_sr = cv2.imread(sr_file)

    if img_gt is None or img_sr is None:
        print(f"Error: Unable to load images {gt_file} or {sr_file}. Check file paths.")
        return

    # Convert to grayscale for Y-channel evaluation
    img_gt_y = cv2.cvtColor(img_gt, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    img_sr_y = cv2.cvtColor(img_sr, cv2.COLOR_BGR2YCrCb)[:, :, 0]

    # Ensure the images are the same size (resize or crop the ground truth)
    h, w = img_sr_y.shape[:2]
    img_gt_y_resized = cv2.resize(img_gt_y, (w, h))  # Resize to match SR image size

    # Calculate metrics
    psnr = compare_psnr(img_gt_y_resized, img_sr_y, data_range=255)
    ssim = compare_ssim(img_gt_y_resized, img_sr_y, data_range=255)
    mae = compute_mae(img_gt_y_resized, img_sr_y)  # Compute MAE

    # Print results
    print(f"PSNR: {psnr:.4f}")
    print(f"SSIM: {ssim:.4f}")
    print(f"MAE: {mae:.4f}")

    # Save results to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(f"PSNR: {psnr:.4f}\n")
            f.write(f"SSIM: {ssim:.4f}\n")
            f.write(f"MAE: {mae:.4f}\n")
        print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PSNR, SSIM, and MAE for specific super-resolved images.")
    parser.add_argument("--gt_file", type=str, required=True, help="Path to the high-resolution ground-truth image.")
    parser.add_argument("--sr_file", type=str, required=True, help="Path to the super-resolved image.")
    parser.add_argument("--scale", type=int, default=4, help="Scaling factor for SR.")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save the evaluation results.")

    args = parser.parse_args()

    calculate_metrics(
        gt_file=args.gt_file,
        sr_file=args.sr_file,
        scale=args.scale,
        output_file=args.output_file
    )
