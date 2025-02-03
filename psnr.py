from math import log10, sqrt
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def PSNR(original, compressed):
    # Resize the original image to match the compressed image size
    original_resized = cv2.resize(original, (compressed.shape[1], compressed.shape[0]))

    mse = np.mean((original_resized - compressed) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal
        # Therefore PSNR has no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def calculate_ssim(original, compressed):
    # Resize the original image to match the compressed image size
    original_resized = cv2.resize(original, (compressed.shape[1], compressed.shape[0]))
    
    # Convert images to grayscale
    original_gray = cv2.cvtColor(original_resized, cv2.COLOR_BGR2GRAY)
    compressed_gray = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)
    
    # Calculate SSIM
    ssim_value = ssim(original_gray, compressed_gray, data_range=compressed_gray.max() - compressed_gray.min())
    return ssim_value

def main():
    original = cv2.imread(r"C:\Users\KIIT\Desktop\medical-image-super-resolution\high.jpg")
    compressed = cv2.imread(r"C:\Users\KIIT\Desktop\medical-image-super-resolution\results\enhanced_low.jpg", 1)
    
    if original is None or compressed is None:
        print("Error loading images")
        exit()
    
    psnr_value = PSNR(original, compressed)
    ssim_value = calculate_ssim(original, compressed)
    
    print(f"PSNR value is {psnr_value} dB")
    print(f"SSIM value is {ssim_value}")

if __name__ == "__main__":
    main()