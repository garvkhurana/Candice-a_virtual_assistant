import torch
import cv2 as cv
import clip
import numpy as np
from PIL import Image
import time
from transformers import DetrImageProcessor, DetrForObjectDetection
from safetensors.torch import load_file

print(" Object Detection System Starting...")

detection_model = None
processor = None
clip_model = None
clip_preprocess = None

HOUSEHOLD_CLASSES = [
    "iPhone", "Android phone", "laptop computer", "wireless earbuds", "AirPods",
    "headphones", "TV remote control", "computer mouse", "wireless mouse",
    "mechanical keyboard", "security camera", "smart speaker", "wifi router",
    "smart light bulb", "power bank", "phone charger", "iPhone charger",
    "laptop charger", "USB-C cable", "Lightning cable", "micro USB cable",
    "wireless charger", "car charger", "kitchen blender", "microwave", "toaster",
    "electric kettle", "air fryer", "hand mixer", "stand mixer", "juicer",
    "food scale", "vacuum cleaner", "electric toothbrush", "hair dryer",
    "nail clipper", "smartwatch", "fitness tracker", "blood pressure monitor",
    "digital thermometer", "weighing scale", "DVD player", "soundbar",
    "bluetooth speaker", "CD player", "printer", "scanner", "calculator",
    "air freshener dispenser", "electric fan"
]

def initialize_models():
    global detection_model, processor, clip_model, clip_preprocess
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f" Using device: {device}")
        
        print(" Loading fine-tuned DETR model...")
        
        try:
            model_path = r"C:\Users\garvk\OneDrive - Bhagwan Parshuram Institute of Technology\Desktop\advance_projects\visual_assistant\classifier_finetuned"
            
            processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
            
            detection_model = DetrForObjectDetection.from_pretrained(model_path)
            detection_model.to(device)
            detection_model.eval()
            
        except Exception as e:
            print(f" Error loading from pretrained path: {e}")
            try:
                detection_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
                processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
                
                detection_model.class_labels_classifier = torch.nn.Linear(
                    detection_model.class_labels_classifier.in_features, 
                    len(HOUSEHOLD_CLASSES)
                )
                
                safetensor_path = r"C:\Users\garvk\OneDrive - Bhagwan Parshuram Institute of Technology\Desktop\advance_projects\visual_assistant\classifier_finetuned\model.safetensors"
                state_dict = load_file(safetensor_path)
                detection_model.load_state_dict(state_dict, strict=False)
                
                detection_model.to(device)
                detection_model.eval()
                
            except Exception as e2:
                print(f" Error loading safetensors: {e2}")
                return False
        
        print(" Fine-tuned DETR model loaded successfully")
        
        try:
            print(" Loading CLIP model...")
            clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
            print(" CLIP model loaded successfully")
        except Exception as e:
            print(f" CLIP not loaded: {e}")
            clip_model = None
        
        return True
        
    except Exception as e:
        print(f" Error loading models: {e}")
        return False

def detect_with_detr(frame):
    try:
        if detection_model is None or processor is None:
            print(" Detection model not loaded")
            return None
            
        device = next(detection_model.parameters()).device
        
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        inputs = processor(images=pil_image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = detection_model(**inputs)
        
        target_sizes = torch.tensor([pil_image.size[::-1]]).to(device)  
        results = processor.post_process_object_detection(
            outputs, 
            target_sizes=target_sizes, 
            threshold=0.3 
        )[0]
        
        detections = []
        
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            score = score.item()
            label = label.item()
            box = box.cpu().numpy()
            
            if label < len(HOUSEHOLD_CLASSES):
                object_name = HOUSEHOLD_CLASSES[label]
            else:
                object_name = f"object_{label}"
            
            bbox = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
            
            frame_center_x = frame.shape[1] // 2
            frame_center_y = frame.shape[0] // 2
            bbox_center_x = (bbox[0] + bbox[2]) / 2
            bbox_center_y = (bbox[1] + bbox[3]) / 2
            
            center_distance = np.sqrt((bbox_center_x - frame_center_x)**2 + 
                                    (bbox_center_y - frame_center_y)**2)
            
            center_bonus = max(0, 1 - (center_distance / (frame.shape[1] * 0.3)))
            adjusted_confidence = score * (1 + center_bonus * 0.3)
            
            detections.append({
                'object_name': object_name,
                'confidence': score,
                'adjusted_confidence': adjusted_confidence,
                'bbox': bbox,
                'detection_type': 'fine_tuned_detr',
                'label_id': label
            })
        
        if detections:
            best_detection = max(detections, key=lambda x: x['adjusted_confidence'])
            return best_detection
            
        return None
        
    except Exception as e:
        print(f" DETR detection error: {e}")
        import traceback
        traceback.print_exc()
        return None

def enhance_with_clip(frame, detr_result=None):
    try:
        if clip_model is None:
            return detr_result
            
        device = next(clip_model.parameters()).device
        
        household_objects = HOUSEHOLD_CLASSES.copy()
        
        image = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        image_input = clip_preprocess(image).unsqueeze(0).to(device)
        
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {obj}") for obj in household_objects]).to(device)
        
        with torch.no_grad():
            logits_per_image, logits_per_text = clip_model(image_input, text_inputs)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        
        best_match_idx = np.argmax(probs[0])
        clip_confidence = float(probs[0][best_match_idx])
        clip_object = household_objects[best_match_idx]
        
        if detr_result and detr_result['confidence'] > 0.6:
            return detr_result
        elif clip_confidence > 0.25 and not detr_result:
            return {
                'object_name': clip_object,
                'confidence': clip_confidence,
                'detection_type': 'clip_enhancement',
                'detr_backup': None
            }
        elif detr_result:
            return detr_result
        else:
            return None
            
    except Exception as e:
        print(f" CLIP enhancement error: {e}")
        return detr_result

def detect_object(frame):
    start_time = time.time()
    
    detr_result = detect_with_detr(frame)
    
    final_result = enhance_with_clip(frame, detr_result) if clip_model else detr_result
    
    processing_time = (time.time() - start_time) * 1000
    if final_result:
        final_result['processing_time_ms'] = processing_time
    
    return final_result

def draw_detection(frame, detection_result):
    if not detection_result:
        return frame
    
    if 'bbox' in detection_result:
        bbox = detection_result['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        
        if detection_result['detection_type'] == 'fine_tuned_detr':
            color = (0, 255, 0)  
        elif detection_result['detection_type'] == 'clip_enhancement':
            color = (255, 0, 0)  
        else:
            color = (0, 255, 255)  
            
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    object_name = detection_result['object_name']
    confidence = detection_result['confidence']
    detection_type = detection_result['detection_type']
    
    text = f"{object_name} ({confidence:.2f})"
    type_text = f"[{detection_type}]"
    
    cv.putText(frame, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv.putText(frame, type_text, (10, 55), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    if 'processing_time_ms' in detection_result:
        time_text = f"Processing: {detection_result['processing_time_ms']:.1f}ms"
        cv.putText(frame, time_text, (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    return frame

def get_detected_object_name(frame):
    
    detection_result = detect_object(frame)
    
    if detection_result and detection_result['confidence'] > 0.5:  
        object_name = detection_result['object_name']
        confidence = detection_result['confidence']
        
        print(f" Detected for LLM: {object_name} (confidence: {confidence:.2f})")
        return object_name
    else:
        print(" No confident object detection for LLM")
        return None

def get_detailed_detection_info(frame):
    detection_result = detect_object(frame)
    
    if detection_result and detection_result['confidence'] > 0.5:
        return {
            'object_name': detection_result['object_name'],
            'confidence': detection_result['confidence'],
            'detection_method': detection_result['detection_type'],
            'processing_time': detection_result.get('processing_time_ms', 0),
            'label_id': detection_result.get('label_id', -1)
        }
    else:
        return None

def main():
    print(" Initializing  Detection System...")
    
    if not initialize_models():
        print(" Failed to initialize models. Exiting...")
        return
    
    # Initialize camera
    print(" Starting camera...")
    cap = cv.VideoCapture(0)
    
    if not cap.isOpened():
        print(" Cannot open camera")
        return
    
    print("JARVIS is ready! Using fine-tuned model for detection. Press 'q' to quit")
    print(f"Trained to detect: {', '.join(HOUSEHOLD_CLASSES[:5])}... and {len(HOUSEHOLD_CLASSES)-5} more objects")
    
    frame_count = 0
    detection_frequency = 15 
    last_detection = None
    detection_stability_count = 0
    stable_detection_threshold = 2
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(" Failed to grab frame")
                break
            
            frame = cv.flip(frame, 1)
            
            frame_count += 1
            current_detection = None
            
            if frame_count % detection_frequency == 0:
                detection_result = detect_object(frame)
                
                if detection_result and detection_result['confidence'] > 0.4: 
                    if (last_detection and 
                        last_detection['object_name'] == detection_result['object_name']):
                        detection_stability_count += 1
                    else:
                        detection_stability_count = 1
                        last_detection = detection_result
                    
                    if detection_stability_count >= stable_detection_threshold:
                        current_detection = detection_result
                        
                        if frame_count % (detection_frequency * 3) == 0:
                            print(f" Stable detection: {detection_result['object_name']} "
                                  f"({detection_result['confidence']:.2f}) "
                                  f"[{detection_result['detection_type']}]")
                else:
                    detection_stability_count = max(0, detection_stability_count - 1)
                    if detection_stability_count == 0:
                        last_detection = None
            
            if current_detection or (last_detection and detection_stability_count > 0):
                detection_to_draw = current_detection or last_detection
                frame = draw_detection(frame, detection_to_draw)
            
            center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
            cv.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 255), 2)
            cv.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 255), 2)
            
            cv.putText(frame, "Fine-tuned JARVIS detecting household objects", 
                      (10, frame.shape[0] - 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv.putText(frame, "Hold object in center area", 
                      (10, frame.shape[0] - 40), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv.putText(frame, "Press Q to quit", 
                      (10, frame.shape[0] - 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv.imshow('JARVIS Fine-tuned Object Detection', frame)
            
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n JARVIS shutting down...")
    
    finally:
        cap.release()
        cv.destroyAllWindows()
        print(" JARVIS offline. Goodbye!")

if __name__ == "__main__":
    main()