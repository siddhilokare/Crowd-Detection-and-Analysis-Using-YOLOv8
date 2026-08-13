# !pip install ultralytics opencv-python pandas numpy tqdm matplotlib

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque

# Parameters
CROWD_THRESHOLD = 3
PROXIMITY_THRESHOLD = 150
MIN_CROWD_DURATION = 1
MODEL_NAME = 'yolov8l.pt'   # Pretrained YOLOv8 large model
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5

# Initialize YOLOv8 model
model = YOLO(MODEL_NAME)

# Video paths
video_path = 'dataset_video.mp4'          # <-- change path if needed
output_path = 'output_enhanced.mp4'

def calculate_centroid(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def are_close(person1, person2, threshold=PROXIMITY_THRESHOLD):
    centroid1 = calculate_centroid(person1['bbox'])
    centroid2 = calculate_centroid(person2['bbox'])
    distance = np.sqrt((centroid1[0] - centroid2[0])**2 + (centroid1[1] - centroid2[1])**2)
    return distance < threshold

def find_crowds(detections):
    crowds = []
    processed = set()

    for i, person in enumerate(detections):
        if i in processed:
            continue

        crowd = [person]
        processed.add(i)
        queue = deque([i])

        while queue:
            current = queue.popleft()
            for j, other in enumerate(detections):
                if j not in processed and are_close(detections[current], other):
                    crowd.append(other)
                    processed.add(j)
                    queue.append(j)

        if len(crowd) >= CROWD_THRESHOLD:
            crowds.append(crowd)

    return crowds

def draw_detections(frame, detections, crowd_detections):
    # Draw individuals
    for person in detections:
        x1, y1, x2, y2 = map(int, person['bbox'])
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{person["conf"]:.2f}', (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw crowds
    for crowd in crowd_detections:
        x1 = min(person['bbox'][0] for person in crowd)
        y1 = min(person['bbox'][1] for person in crowd)
        x2 = max(person['bbox'][2] for person in crowd)
        y2 = max(person['bbox'][3] for person in crowd)

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
        cv2.putText(frame, f'Crowd: {len(crowd)}', (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return frame

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error opening video file")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:  # fallback if metadata missing
        fps = 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    crowd_events = []
    crowd_start_times = {}
    crowd_metadata = {}
    frame_count = 0

    for _ in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time = frame_count / fps

        results = model(frame,
                       classes=[0],
                       conf=CONFIDENCE_THRESHOLD,
                       iou=IOU_THRESHOLD,
                       imgsz=640,
                       verbose=False)

        detections = []
        for result in results:
            for box in result.boxes:
                conf = float(box.conf)
                bbox = box.xyxy[0].cpu().numpy()
                detections.append({'bbox': bbox, 'conf': conf})

        current_crowds = find_crowds(detections)

        current_crowd_ids = []
        for crowd in current_crowds:
            avg_x = sum(p['bbox'][0] for p in crowd) / len(crowd)
            avg_y = sum(p['bbox'][1] for p in crowd) / len(crowd)
            crowd_id = f"{int(avg_x)}_{int(avg_y)}"
            current_crowd_ids.append(crowd_id)

            if crowd_id not in crowd_start_times:
                crowd_start_times[crowd_id] = current_time
                crowd_metadata[crowd_id] = {
                    'location_x': int(avg_x),
                    'location_y': int(avg_y),
                    'max_people': len(crowd)
                }
            else:
                crowd_metadata[crowd_id]['max_people'] = max(
                    crowd_metadata[crowd_id]['max_people'], len(crowd))

        # Check for ended crowds
        ended_crowds = set(crowd_start_times.keys()) - set(current_crowd_ids)
        for crowd_id in ended_crowds:
            start_time = crowd_start_times[crowd_id]
            duration = current_time - start_time
            if duration >= MIN_CROWD_DURATION:
                crowd_events.append({
                    'start_time': start_time,
                    'end_time': current_time,
                    'duration': duration,
                    'max_people': crowd_metadata[crowd_id]['max_people'],
                    'location_x': crowd_metadata[crowd_id]['location_x'],
                    'location_y': crowd_metadata[crowd_id]['location_y']
                })
            del crowd_start_times[crowd_id]
            del crowd_metadata[crowd_id]

        frame_with_detections = draw_detections(frame.copy(), detections, current_crowds)
        out.write(frame_with_detections)

    cap.release()
    out.release()
    return crowd_events

# Run detection
print("Starting enhanced detection...")
crowd_events = process_video(video_path, output_path)

if crowd_events:
    events_df = pd.DataFrame(crowd_events)
    print("\nEnhanced Detection Results:")
    print(f"Total crowd events: {len(events_df)}")
    print(events_df.head())

    events_df.to_csv('enhanced_crowd_events.csv', index=False)
    print("\nSaved enhanced results to 'enhanced_crowd_events.csv'")
    print("Annotated video saved to 'output_enhanced.mp4'")

    # Show sample frame
    cap = cv2.VideoCapture(output_path)
    ret, frame = cap.read()
    if ret:
        plt.figure(figsize=(16, 10))
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("Enhanced Detection Sample")
        plt.show()
    cap.release()
else:
    print("No crowd events detected with current settings.")