import cv2
import torch
import numpy as np
from ultralytics import YOLO

#label the name tag


#train YOLO model


#identify the frames with name tag
def process_video(video_path, model_path):
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    staff_frames = []
    coordinates = {}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        results = model(frame)
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                class_id = int(box.cls[0])
                
                if class_id == 0 and conf > 0.5:  # Assuming 0 is the class for 'staff name tag'
                    staff_frames.append(frame_count)
                    coordinates[frame_count] = (x1.item(), y1.item(), x2.item(), y2.item())
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"Staff ({conf:.2f})", (int(x1), int(y1) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Optional: Display processed frames
        cv2.imshow('Staff Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return staff_frames, coordinates

video_path = "./sample.mp4"
model_path = "yolov8_best.pt"  
staff_frames, staff_coordinates = process_video(video_path, model_path)

print("Frames with staff detected:", staff_frames)
print("Staff coordinates:", staff_coordinates)
