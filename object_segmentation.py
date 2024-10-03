import argparse
from segment_anything import SamPredictor, sam_model_registry
from groundingdino.util.inference import load_model, load_image, predict
from GroundingDINO.groundingdino.util import box_ops
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt

def show_mask(mask, image):
    color = np.array([1.0, 0.0, 0.0, 0.5])  # Red color with 50% opacity
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image * 255).astype(np.uint8)).convert("RGBA")

    return Image.alpha_composite(annotated_frame_pil, mask_image_pil)

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load SAM
    sam_checkpoint = "/Users/apple/Documents/CS/avataar_assignment/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    predictor = SamPredictor(sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=device))

    # Load GroundingDINO
    groundingdino_config = "/Users/apple/Documents/CS/avataar_assignment/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    groundingdino_checkpoint = "/Users/apple/Documents/CS/avataar_assignment/Grounding_Dino_SWINT_OGC.pth"
    groundingdino_model = load_model(groundingdino_config, groundingdino_checkpoint).to(device=device)

    # Load image
    src, img = load_image(args.image)

    # Predict with GroundingDINO
    boxes, _, _ = predict(
        model=groundingdino_model,
        image=img,
        caption=args.class_name,
        box_threshold=0.3,
        text_threshold=0.25,
        device=device
    )

    # Predict with SAM
    predictor.set_image(src)
    H, W, _ = src.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    new_boxes = predictor.transform.apply_boxes_torch(boxes_xyxy, src.shape[:2]).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=new_boxes,
        multimask_output=False,
    )

    # Get the binary mask
    binary_mask = masks[0][0].cpu().numpy()

    if args.mask_only:
        # Save only the mask
        plt.figure(figsize=(10, 10))
        plt.imshow(binary_mask, cmap='gray')
        plt.axis('off')
        plt.savefig(args.output, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        # Create final image with red mask
        final_image = show_mask(binary_mask, src)
        final_image.save(args.output)

    # Display the results
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    plt.suptitle(f"Segmentation Results for '{args.class_name}'", fontsize=16)
    
    axes[0].imshow(src)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(binary_mask, cmap='gray')
    axes[1].set_title("Segmentation Mask")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Segmentation Script")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--class", dest="class_name", type=str, required=True, help="Class name to segment")
    parser.add_argument("--output", type=str, required=True, help="Path to save output image")
    parser.add_argument("--mask-only", action="store_true", help="Save only the binary mask")
    
    args = parser.parse_args()
    main(args)