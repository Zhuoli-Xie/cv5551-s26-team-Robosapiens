import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import os

# ============================================
# 1. LOAD GROUNDING DINO (for bounding box)
# ============================================
model_id = "IDEA-Research/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)

# ============================================
# 2. LOAD SAM (for segmentation)
# ============================================
# Download checkpoint if not exists
sam_checkpoint = "sam_vit_b_01ec64.pth"
if not os.path.exists(sam_checkpoint):
    print("📥 no find SAM ViT-B checkpoint...")
    

# Load SAM model
from segment_anything import sam_model_registry, SamPredictor

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Using device: {device}")

sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
sam = sam.to(device)
sam_predictor = SamPredictor(sam)

# ============================================
# 3. LOAD IMAGE
# ============================================
image_path = "cam2.png"  # Change this!
image = Image.open(image_path)
if image.mode != 'RGB':
    image = image.convert('RGB')

# Convert to OpenCV format (for display and SAM)
image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# ============================================
# 4. DETECT PHONE with Grounding DINO
# ============================================
text = "black phone."
inputs = processor(images=image, text=text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs=outputs,
    input_ids=inputs.input_ids,
    threshold=0.4,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]],
    text_labels=True
)

# ============================================
# 5. SEGMENT with SAM using bounding box prompt
# ============================================
# Set the image in SAM predictor
sam_predictor.set_image(image_cv)

# Process each detection
for result in results:
    for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
        if label == "black phone":
            # Convert box to numpy array [x1, y1, x2, y2]
            box_np = box.numpy()
            
            # Predict mask with SAM
            masks, scores, logits = sam_predictor.predict(
                box=box_np,
                multimask_output=False  # Single best mask
            )
            
            # Get the mask (boolean array)
            mask = masks[0]  # Shape: (H, W)
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            print(f"\n📱 Phone detected with confidence: {score:.3f}")
            print(f"   Bounding box: {box_np}")
            print(f"   Mask shape: {mask.shape}")
            
            # ============================================
            # 6. VISUALIZE RESULTS
            # ============================================
            # Create a copy for visualization
            vis_image = image_cv.copy()
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, box_np)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Create colored mask overlay
            # Red color for mask with transparency
            color_mask = np.zeros_like(vis_image)
            color_mask[:, :, 2] = mask_uint8  # Red channel
            
            # Blend mask with image
            alpha = 0.5
            vis_image = cv2.addWeighted(vis_image, 1, color_mask, alpha, 0)
            
            # Add label
            cv2.putText(vis_image, f"Phone: {score:.2f}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # ============================================
            # 7. CREATE EXTRACTED PHONE (masked background)
            # ============================================
            # Create image with only the phone (background black)
            extracted_phone = image_cv.copy()
            extracted_phone[mask == 0] = 0  # Black out background
            
            # Display all images
            cv2.namedWindow("Detection + Mask", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Detection + Mask", 800, 600)
            cv2.imshow("Detection + Mask", vis_image)
            
            cv2.namedWindow("Extracted Phone (Background Removed)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Extracted Phone (Background Removed)", 800, 600)
            cv2.imshow("Extracted Phone (Background Removed)", extracted_phone)
            
            # Optional: Show just the mask
            cv2.namedWindow("Phone Mask", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Phone Mask", 800, 600)
            cv2.imshow("Phone Mask", mask_uint8)
            
            print("\n📺 Press any key to close windows...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # ============================================
            # 8. SAVE RESULTS
            # ============================================
            cv2.imwrite("data/phone_with_mask.jpg", vis_image)
            cv2.imwrite("data/phone_extracted.jpg", extracted_phone)
            cv2.imwrite("data/phone_mask_only.jpg", mask_uint8)
            print("\n💾 Saved results:")
            print("   - phone_with_mask.jpg (bounding box + mask overlay)")
            print("   - phone_extracted.jpg (phone only, black background)")
            print("   - phone_mask_only.jpg (binary mask only)")
            
            break  # Only process first phone detection
