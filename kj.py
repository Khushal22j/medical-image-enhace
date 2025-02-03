import argparse
import cv2
import glob
import os
import logging
from tqdm import tqdm
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('super_resolution.log'),
                              logging.StreamHandler()])

def validate_paths(input_path, output_path):
    """Validate input and output paths."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path {input_path} does not exist")
    os.makedirs(output_path, exist_ok=True)
    logging.info(f"Output directory validated/created: {output_path}")

def load_model(args):
    """Load appropriate model based on arguments."""
    model_configs = {
        'RealESRGAN_x4plus': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
        'RealESRGAN_x4plus_anime_6B': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4),
        'RealESRGAN_x2plus': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2),
        'realesr-animevideov3': SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
    }
    
    model_name = args.model_name.split('.')[0]
    if model_name not in model_configs:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model_configs[model_name], 4 if model_name != 'RealESRGAN_x2plus' else 2

def process_image(upsampler, img, args):
    """Process a single image with error handling."""
    try:
        output, _ = upsampler.enhance(img, outscale=args.outscale)
        return output
    except RuntimeError as error:
        logging.error(f"Processing error: {error}")
        if 'CUDA out of memory' in str(error):
            logging.warning("Try reducing tile size (--tile) or using CPU mode")
        return None

def main():
    """Enhanced inference demo for Real-ESRGAN with medical imaging improvements."""
    parser = argparse.ArgumentParser(description='Medical Image Super-Resolution Tool')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Input image or folder (supports common medical formats)')
    parser.add_argument('-n', '--model_name', type=str, default='RealESRGAN_x4plus',
                        choices=['RealESRGAN_x4plus', 'RealESRNet_x4plus', 'RealESRGAN_x4plus_anime_6B',
                                 'RealESRGAN_x2plus', 'realesr-animevideov3'],
                        help='Model selection for different use cases')
    parser.add_argument('-o', '--output', type=str, default='results',
                        help='Output folder for processed images')
    parser.add_argument('-s', '--outscale', type=float, default=4,
                        help='Upsampling scale factor (2-4 depending on model)')
    parser.add_argument('--suffix', type=str, default='superres',
                        help='Suffix for output filenames')
    parser.add_argument('-t', '--tile', type=int, default=400,
                        help='Tile size for memory optimization (0 for no tiling)')
    parser.add_argument('--tile_pad', type=int, default=20,
                        help='Padding around tiles to avoid edge artifacts')
    parser.add_argument('--pre_pad', type=int, default=0,
                        help='Pre-padding size for border handling')
    parser.add_argument('--fp32', action='store_true',
                        help='Use FP32 precision (recommended for CPU execution)')
    parser.add_argument('--gpu_id', type=int, default=None,
                        help='GPU device ID (None for automatic selection)')
    parser.add_argument('--retain_original_bitdepth', action='store_true',
                        help='Maintain original image bit depth in output')

    args = parser.parse_args()

    # Validate paths first
    validate_paths(args.input, args.output)

    # Model initialization
    model, netscale = load_model(args)
    
    # Model path handling
    model_path = os.path.join('experiments/pretrained_models', f'{args.model_name}.pth')
    if not os.path.isfile(model_path):
        model_path = os.path.join('realesrgan/weights', f'{args.model_name}.pth')
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    # Initialize upsampler with FP32 (disabling FP16)
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=False,  # Disable FP16 (use FP32)
        gpu_id=args.gpu_id
    )

    # File handling
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))
        paths = [p for p in paths if os.path.splitext(p)[1].lower() in 
                ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.dcm']]

    # Processing loop with progress bar
    success_count = 0
    for path in tqdm(paths, desc="Processing images"):
        try:
            imgname = os.path.splitext(os.path.basename(path))[0]
            logging.info(f"Processing {imgname}")
            
            # Read image with medical format support
            if path.lower().endswith('.dcm'):
                import pydicom
                dcm = pydicom.dcmread(path)
                img = dcm.pixel_array
                if hasattr(dcm, 'PhotometricInterpretation') and dcm.PhotometricInterpretation == 'MONOCHROME1':
                    img = cv2.bitwise_not(img)  # Invert if needed
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            else:
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

            if img is None:
                logging.warning(f"Failed to read {path}")
                continue

            # Process image
            output = process_image(upsampler, img, args)  # Removed GFPGAN for medical use
            
            if output is not None:
                # Handle output bit depth
                if args.retain_original_bitdepth:
                    output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                
                # Save result
                ext = 'png' if args.retain_original_bitdepth else 'tiff'
                save_path = os.path.join(args.output, f"{imgname}_{args.suffix}.{ext}")
                cv2.imwrite(save_path, output)
                success_count += 1

        except Exception as e:
            logging.error(f"Error processing {path}: {str(e)}")

    logging.info(f"Processing complete. Successfully processed {success_count}/{len(paths)} images")

if __name__ == '__main__':
    main()
