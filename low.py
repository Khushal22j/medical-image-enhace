import os
import cv2
import argparse


def create_low_resolution_images(hr_dir, lr_dir, scale_factor):
    """
    Generate low-resolution images by downscaling high-resolution images.

    Args:
        hr_dir (str): Path to the high-resolution image directory.
        lr_dir (str): Path to save low-resolution images.
        scale_factor (int): Downscaling factor (e.g., 4 for 1/4 resolution).
    """
    # Ensure the output directory exists
    os.makedirs(lr_dir, exist_ok=True)

    # Process each image in the HR directory
    for filename in os.listdir(hr_dir):
        hr_path = os.path.join(hr_dir, filename)

        # Check if the file is a valid image
        if not os.path.isfile(hr_path):
            print(f"Skipping non-file: {hr_path}")
            continue

        # Read the HR image
        img = cv2.imread(hr_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to read image: {hr_path}")
            continue

        # Compute new dimensions
        height, width = img.shape
        new_height = height // scale_factor
        new_width = width // scale_factor

        # Resize the image
        lr_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Save the LR image
        lr_path = os.path.join(lr_dir, filename)
        cv2.imwrite(lr_path, lr_img)
        print(f"Saved LR image: {lr_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate low-resolution images from high-resolution images.")
    parser.add_argument("--hr_dir", type=str, required=True, help="Path to the high-resolution image directory")
    parser.add_argument("--lr_dir", type=str, required=True, help="Path to save low-resolution images")
    parser.add_argument("--scale_factor", type=int, default=4, help="Downscaling factor (default: 4)")

    args = parser.parse_args()

    # Generate low-resolution images
    create_low_resolution_images(
        hr_dir=args.hr_dir,
        lr_dir=args.lr_dir,
        scale_factor=args.scale_factor,
    )
