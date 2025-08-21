from transformers import pipeline
import torch
import cv2 as cv
import clip
import numpy as np
from PIL import Image
import time

print("🤖 JARVIS Object Detection System Starting...")

# Global models
detection_model = None
clip_model = None
clip_preprocess = None

def initialize_models():
    """Initialize both detection and CLIP models"""
    global detection_model, clip_model, clip_preprocess
    
    try:
        # Check device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔥 Using device: {device}")
        
        # Load object detection model
        print("⚡ Loading detection model...")
        if detection_model is None:
            detection_model = pipeline("object-detection", model="facebook/detr-resnet-50")
        print("✅ DETR model loaded successfully")
        
        # Load CLIP model for enhanced recognition
        print("🎨 Loading CLIP model...")
        if clip_model is None:
            clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        print("✅ CLIP model loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

def detect_with_detr(frame):
    """Primary detection using DETR - excludes humans"""
    try:
        if detection_model is None:
            print("⚠️ Detection model not loaded")
            return None
            
        # Objects to ignore (humans and body parts)
        excluded_classes = ['person', 'face', 'hand', 'arm', 'leg', 'head', 'body']
        
        # Convert OpenCV frame (BGR) to PIL Image (RGB)
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Run detection
        results = detection_model(pil_image)
        
        detections = []
        for detection in results:
            object_name = detection['label']
            confidence = detection['score']
            box = detection['box']  # Format: {'xmin': x, 'ymin': y, 'xmax': x, 'ymax': y}
            
            # Skip human detections
            if any(excluded_class in object_name.lower() for excluded_class in excluded_classes):
                continue
            
            # Convert box format to standard bbox [x1, y1, x2, y2]
            bbox = [box['xmin'], box['ymin'], box['xmax'], box['ymax']]
            
            # Focus on objects in center area (where hands hold objects)
            frame_center_x = frame.shape[1] // 2
            frame_center_y = frame.shape[0] // 2
            bbox_center_x = (bbox[0] + bbox[2]) / 2
            bbox_center_y = (bbox[1] + bbox[3]) / 2
            
            # Calculate distance from center
            center_distance = np.sqrt((bbox_center_x - frame_center_x)**2 + 
                                    (bbox_center_y - frame_center_y)**2)
            
            # Prefer objects closer to center (likely in hands)
            center_bonus = max(0, 1 - (center_distance / (frame.shape[1] * 0.3)))
            adjusted_confidence = confidence * (1 + center_bonus * 0.3)
            
            detections.append({
                'object_name': object_name,
                'confidence': confidence,
                'adjusted_confidence': adjusted_confidence,
                'bbox': bbox,
                'detection_type': 'detr'
            })
        
        # Return highest adjusted confidence detection
        if detections:
            best_detection = max(detections, key=lambda x: x['adjusted_confidence'])
            return best_detection
            
        return None
        
    except Exception as e:
        print(f"❌ DETR detection error: {e}")
        return None

def enhance_with_clip(frame, detr_result=None):
    """Enhanced recognition using CLIP for household objects"""
    try:
        if clip_model is None:
            print("⚠️ CLIP model not loaded")
            return detr_result
            
        # Check device for CLIP
        device = next(clip_model.parameters()).device
        
        # Household objects that JARVIS should recognize
        household_objects = [
            "smartphone", "mobile phone", "iPhone", "Android phone",
            "laptop computer", "tablet", "iPad",
            "wireless earbuds", "AirPods", "headphones",
            "TV remote control", "gaming controller", "Xbox controller", "PlayStation controller",
            "computer mouse", "keyboard", "webcam",
            "smart speaker", "Amazon Echo", "Google Home",
            "router", "modem", "wifi device",
            "power bank", "phone charger", "USB cable",
            "kitchen blender", "coffee maker", "microwave",
            "vacuum cleaner", "electric toothbrush",
            "smartwatch", "fitness tracker"
        ]
        
        # Convert frame to PIL Image
        image = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        image_input = clip_preprocess(image).unsqueeze(0).to(device)
        
        # Create text inputs
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {obj}") for obj in household_objects]).to(device)
        
        # Calculate similarities
        with torch.no_grad():
            logits_per_image, logits_per_text = clip_model(image_input, text_inputs)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        
        # Get best match
        best_match_idx = np.argmax(probs[0])
        clip_confidence = float(probs[0][best_match_idx])
        clip_object = household_objects[best_match_idx]
        
        # Decide which detection to use
        if detr_result and detr_result['confidence'] > 0.7:
            # DETR is confident, use it
            return detr_result
        elif clip_confidence > 0.3:
            # CLIP has a reasonable match
            return {
                'object_name': clip_object,
                'confidence': clip_confidence,
                'detection_type': 'clip',
                'detr_backup': detr_result
            }
        elif detr_result:
            # Use DETR as fallback
            return detr_result
        else:
            # No good detection
            return {
                'object_name': 'unknown_object',
                'confidence': 0.1,
                'detection_type': 'unknown'
            }
            
    except Exception as e:
        print(f"❌ CLIP enhancement error: {e}")
        return detr_result

def detect_object(frame):
    """Main detection function - DETR primary, CLIP enhancement available"""
    start_time = time.time()
    
    # Use DETR for primary detection
    detr_result = detect_with_detr(frame)
    
    # Optional: Enable CLIP enhancement
    # final_result = enhance_with_clip(frame, detr_result)
    final_result = detr_result  # Using DETR only for now
    
    # Add timing info
    processing_time = (time.time() - start_time) * 1000
    if final_result:
        final_result['processing_time_ms'] = processing_time
    
    return final_result

def draw_detection(frame, detection_result):
    """Draw detection results on frame"""
    if not detection_result:
        return frame
    
    # Draw bounding box if available
    if 'bbox' in detection_result:
        bbox = detection_result['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw text
    object_name = detection_result['object_name']
    confidence = detection_result['confidence']
    detection_type = detection_result['detection_type']
    
    text = f"{object_name} ({confidence:.2f}) [{detection_type}]"
    cv.putText(frame, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Show processing time
    if 'processing_time_ms' in detection_result:
        time_text = f"Processing: {detection_result['processing_time_ms']:.1f}ms"
        cv.putText(frame, time_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    return frame

def get_detected_object_name(frame):
    """
    Main function to get object name for LLM integration
    Returns: string object name or None
    """
    detection_result = detect_object(frame)
    
    if detection_result and detection_result['confidence'] > 0.4:
        object_name = detection_result['object_name']
        confidence = detection_result['confidence']
        
        print(f"🎯 Detected for LLM: {object_name} (confidence: {confidence:.2f})")
        return object_name
    else:
        print("❌ No confident object detection for LLM")
        return None

def get_detailed_detection_info(frame):
    """
    Get full detection details for advanced LLM context
    Returns: dict with object info or None
    """
    detection_result = detect_object(frame)
    
    if detection_result and detection_result['confidence'] > 0.4:
        return {
            'object_name': detection_result['object_name'],
            'confidence': detection_result['confidence'],
            'detection_method': detection_result['detection_type'],
            'processing_time': detection_result.get('processing_time_ms', 0)
        }
    else:
        return None

def main():
    """Main function to test the detection system"""
    print("🚀 Initializing JARVIS Detection System...")
    
    # Initialize models
    if not initialize_models():
        print("💥 Failed to initialize models. Exiting...")
        return
    
    # Initialize camera
    print("📹 Starting camera...")
    cap = cv.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    print("✅ JARVIS is ready! Continuous detection active. Press 'q' to quit")
    
    # Variables for continuous detection
    frame_count = 0
    detection_frequency = 30  # Detect every 30 frames for better performance
    last_detection = None
    detection_stability_count = 0
    stable_detection_threshold = 2
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv.flip(frame, 1)
            
            # Continuous detection (every N frames for performance)
            frame_count += 1
            current_detection = None
            
            if frame_count % detection_frequency == 0:
                detection_result = detect_object(frame)
                
                if detection_result and detection_result['confidence'] > 0.5:
                    # Check if detection is stable (same object detected multiple times)
                    if (last_detection and 
                        last_detection['object_name'] == detection_result['object_name']):
                        detection_stability_count += 1
                    else:
                        detection_stability_count = 1
                        last_detection = detection_result
                    
                    # Only show stable detections to avoid flickering
                    if detection_stability_count >= stable_detection_threshold:
                        current_detection = detection_result
                        
                        # Print detection info (less frequent to avoid spam)
                        if frame_count % (detection_frequency * 3) == 0:
                            print(f"🎯 Stable detection: {detection_result['object_name']} "
                                  f"({detection_result['confidence']:.2f}) "
                                  f"[{detection_result['detection_type']}]")
                else:
                    # Reset if no good detection
                    detection_stability_count = max(0, detection_stability_count - 1)
                    if detection_stability_count == 0:
                        last_detection = None
            
            # Draw detection if we have a stable one
            if current_detection or (last_detection and detection_stability_count > 0):
                detection_to_draw = current_detection or last_detection
                frame = draw_detection(frame, detection_to_draw)
            
            # Draw center crosshair to help with object placement
            center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
            cv.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 255), 2)
            cv.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 255), 2)
            
            # Display instructions
            cv.putText(frame, "Hold object in center area - JARVIS detecting continuously", 
                      (10, frame.shape[0] - 40), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv.putText(frame, "Press Q to quit", 
                      (10, frame.shape[0] - 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show frame
            cv.imshow('JARVIS Object Detection', frame)
            
            # Check for quit
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n🛑 JARVIS shutting down...")
    
    finally:
        # Cleanup
        cap.release()
        cv.destroyAllWindows()
        print("👋 JARVIS offline. Goodbye!")

if __name__ == "__main__":
    main()