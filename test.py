import os
import argparse
import torch
import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet
from tqdm import tqdm


def load_model(model_path, scale):
    """Load the trained model."""
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model


def enhance_image(model, img_lq):
    """Enhance a single image using the trained model."""
    with torch.no_grad():
        img_lq = torch.from_numpy(img_lq).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_lq = img_lq.to(torch.device("cpu"))  # Change to "cuda" if GPU is available
        output = model(img_lq)
        output = output.squeeze().permute(1, 2, 0).cpu().numpy() * 255.0
        output = output.clip(0, 255).astype('uint8')
    return output


def process_single_image(model, input_path, output_dir):
    """Process a single image."""
    img_lq = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img_lq is None:
        print(f"Failed to read image: {input_path}")
        return
    output = enhance_image(model, img_lq)

    # Save the enhanced image
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"enhanced_{os.path.basename(input_path)}")
    cv2.imwrite(output_path, output)
    print(f"Enhanced image saved to: {output_path}")


def process_multiple_images(model, input_dir, output_dir):
    """Process all images in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    test_images = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    if not test_images:
        print(f"No images found in directory: {input_dir}")
        return

    for img_path in tqdm(test_images, desc="Processing images"):
        img_lq = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_lq is None:
            print(f"Failed to read image: {img_path}")
            continue
        output = enhance_image(model, img_lq)

        # Save the enhanced image
        output_path = os.path.join(output_dir, f"enhanced_{os.path.basename(img_path)}")
        cv2.imwrite(output_path, output)
        print(f"Enhanced image saved to: {output_path}")


def main(args):
    # Load the trained model
    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, args.scale)

    if os.path.isfile(args.input_path):  # Single image
        process_single_image(model, args.input_path, args.output_dir)
    elif os.path.isdir(args.input_path):  # Directory of images
        process_multiple_images(model, args.input_path, args.output_dir)
    else:
        print(f"Invalid input path: {args.input_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained Real-ESRGAN model on low-resolution images")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input image or directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save enhanced images")
    parser.add_argument("--scale", type=int, default=4, help="Scale factor used during training")
    args = parser.parse_args()

    main(args)
