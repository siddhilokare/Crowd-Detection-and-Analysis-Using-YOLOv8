#!/usr/bin/env python3

"""
Market Crowd Monitoring System
--------------------------------
YOLOv8 + ByteTrack + Perspective Normalization + BFS
+ Temporal Crowd Event Detection

Outputs:
    1. Annotated MP4 video
    2. CSV containing detected crowd events

Important:
- The perspective calibration coordinates below are initial estimates
  based on the provided market-scene screenshot.
- WORLD_POINTS are normalized coordinates (0-100), NOT metres.
- Tune CROWD_THRESHOLD, PROXIMITY_THRESHOLD, and MIN_CROWD_DURATION
  against the actual video.
"""

import cv2
import numpy as np
import pandas as pd

from ultralytics import YOLO
from tqdm import tqdm
from collections import deque


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "yolov8l.pt"

CONFIDENCE_THRESHOLD = 0.30
IOU_THRESHOLD = 0.50
PERSON_CLASS = 0
IMAGE_SIZE = 640

# Crowd settings
CROWD_THRESHOLD = 4
PROXIMITY_THRESHOLD = 8.0          # normalized ground-plane units
MIN_CROWD_DURATION = 2.0           # seconds
CROWD_GRACE_PERIOD = 1.0           # seconds
MIN_JACCARD_SIMILARITY = 0.40

# Input/output files
VIDEO_PATH = "dataset_video.mp4"
OUTPUT_VIDEO_PATH = "market_crowd_tracking.mp4"
OUTPUT_CSV_PATH = "market_crowd_events.csv"


# ============================================================
# PERSPECTIVE CALIBRATION
# ============================================================

# Initial points for the provided market scene.
# Replace these with better points from an actual video frame
# if needed.
IMAGE_POINTS = np.float32([
    [500, 170],      # Far-left
    [1250, 170],     # Far-right
    [1650, 900],     # Near-right
    [180, 900],      # Near-left
])

# Normalized ground plane.
# These are NOT real-world metres.
WORLD_POINTS = np.float32([
    [0, 0],
    [100, 0],
    [100, 100],
    [0, 100],
])

HOMOGRAPHY = cv2.getPerspectiveTransform(
    IMAGE_POINTS,
    WORLD_POINTS
)


# ============================================================
# YOLO MODEL
# ============================================================

def load_model():
    print(f"Loading {MODEL_NAME} ...")
    model = YOLO(MODEL_NAME)
    print("Model loaded successfully.")
    return model


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def image_to_ground(point):
    """Convert an image pixel point to normalized ground coordinates."""
    point = np.asarray(point, dtype=np.float32).reshape(1, 1, 2)
    transformed = cv2.perspectiveTransform(point, HOMOGRAPHY)
    return transformed[0, 0]


def get_foot_point(bbox):
    """Return the bottom-center point of a person bounding box."""
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2.0, y2], dtype=np.float32)


def calculate_distance(person1, person2):
    """Euclidean distance between two people on normalized ground plane."""
    return float(
        np.linalg.norm(
            person1["ground_point"] - person2["ground_point"]
        )
    )


def are_close(person1, person2):
    """Check whether two people are spatially close."""
    return calculate_distance(person1, person2) <= PROXIMITY_THRESHOLD


# ============================================================
# CROWD DETECTION
# ============================================================

def find_crowds(detections):
    """
    Build a proximity graph over detected people and use BFS
    to find connected components of sufficient size.
    """
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
                if j in processed:
                    continue

                if are_close(detections[current], other):
                    crowd.append(other)
                    processed.add(j)
                    queue.append(j)

        if len(crowd) >= CROWD_THRESHOLD:
            crowds.append(crowd)

    return crowds


def jaccard_similarity(set_a, set_b):
    """Compute Jaccard similarity between two sets of person IDs."""
    union = set_a.union(set_b)
    if not union:
        return 0.0

    intersection = set_a.intersection(set_b)
    return len(intersection) / len(union)


def match_crowd(current_members, active_crowds):
    """
    Match a current crowd to an existing crowd using overlap
    of persistent person IDs.
    """
    best_id = None
    best_score = 0.0

    for crowd_id, crowd_data in active_crowds.items():
        previous_members = set(crowd_data["members"])
        score = jaccard_similarity(
            set(current_members),
            previous_members
        )

        if score > best_score:
            best_score = score
            best_id = crowd_id

    if best_score >= MIN_JACCARD_SIMILARITY:
        return best_id

    return None


# ============================================================
# DRAWING
# ============================================================

def draw_detections(frame, detections, crowds):
    """Draw person detections and crowd bounding boxes."""
    # Individual people
    for person in detections:
        x1, y1, x2, y2 = map(int, person["bbox"])
        person_id = person["id"]
        confidence = person["conf"]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        foot_x, foot_y = map(int, person["foot_point"])

        cv2.circle(
            frame,
            (foot_x, foot_y),
            4,
            (255, 0, 0),
            -1,
        )

        label = f"ID:{person_id} {confidence:.2f}"

        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    # Crowd groups
    for crowd in crowds:
        x1 = min(person["bbox"][0] for person in crowd)
        y1 = min(person["bbox"][1] for person in crowd)
        x2 = max(person["bbox"][2] for person in crowd)
        y2 = max(person["bbox"][3] for person in crowd)

        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 0, 255),
            3,
        )

        label = f"CROWD: {len(crowd)}"

        cv2.putText(
            frame,
            label,
            (int(x1), max(30, int(y1) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return frame


# ============================================================
# MAIN VIDEO PROCESSING
# ============================================================

def process_video(model, input_path, output_path):
    """Run detection, tracking, crowd analysis, and video writing."""
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise FileNotFoundError(
            f"Could not open input video: {input_path}"
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    if fps <= 0:
        fps = 30.0

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print("\nVideo information")
    print("-----------------")
    print(f"Width       : {width}")
    print(f"Height      : {height}")
    print(f"FPS         : {fps:.2f}")
    print(f"Total frames: {total_frames}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height),
    )

    if not out.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not create output video: {output_path}"
        )

    active_crowds = {}
    crowd_events = []
    next_crowd_id = 1
    frame_count = 0

    try:
        for _ in tqdm(
            range(total_frames),
            desc="Processing market video"
        ):
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            current_time = frame_count / fps

            # YOLO + ByteTrack
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[PERSON_CLASS],
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                imgsz=IMAGE_SIZE,
                verbose=False,
            )

            detections = []

            for result in results:
                boxes = result.boxes

                if boxes is None or boxes.id is None:
                    continue

                track_ids = (
                    boxes.id
                    .int()
                    .cpu()
                    .tolist()
                )

                bboxes = (
                    boxes.xyxy
                    .cpu()
                    .numpy()
                )

                confidences = (
                    boxes.conf
                    .cpu()
                    .numpy()
                )

                for track_id, bbox, confidence in zip(
                    track_ids,
                    bboxes,
                    confidences,
                ):
                    foot_point = get_foot_point(bbox)
                    ground_point = image_to_ground(foot_point)

                    detections.append({
                        "id": int(track_id),
                        "bbox": bbox,
                        "conf": float(confidence),
                        "foot_point": foot_point,
                        "ground_point": ground_point,
                    })

            # Spatial crowd detection
            current_crowds = find_crowds(detections)
            seen_crowd_ids = set()

            # Update/create crowd states
            for crowd in current_crowds:
                current_members = frozenset(
                    person["id"] for person in crowd
                )

                existing_id = match_crowd(
                    current_members,
                    active_crowds,
                )

                if existing_id is not None:
                    crowd_id = existing_id
                    crowd_data = active_crowds[crowd_id]

                    crowd_data["members"] = set(current_members)
                    crowd_data["last_seen"] = current_time
                    crowd_data["max_people"] = max(
                        crowd_data["max_people"],
                        len(crowd),
                    )

                    seen_crowd_ids.add(crowd_id)

                else:
                    crowd_id = next_crowd_id
                    next_crowd_id += 1

                    ground_points = np.array([
                        person["ground_point"]
                        for person in crowd
                    ])

                    average_position = ground_points.mean(axis=0)

                    active_crowds[crowd_id] = {
                        "members": set(current_members),
                        "start_time": current_time,
                        "last_seen": current_time,
                        "max_people": len(crowd),
                        "location_x": float(average_position[0]),
                        "location_y": float(average_position[1]),
                    }

                    seen_crowd_ids.add(crowd_id)

            # Close crowds that have been absent beyond the grace period
            ended_crowds = []

            for crowd_id, crowd_data in list(
                active_crowds.items()
            ):
                if crowd_id in seen_crowd_ids:
                    continue

                missing_time = (
                    current_time
                    - crowd_data["last_seen"]
                )

                if missing_time > CROWD_GRACE_PERIOD:
                    start_time = crowd_data["start_time"]
                    end_time = crowd_data["last_seen"]
                    duration = end_time - start_time

                    if duration >= MIN_CROWD_DURATION:
                        crowd_events.append({
                            "crowd_id": crowd_id,
                            "start_time": round(start_time, 2),
                            "end_time": round(end_time, 2),
                            "duration": round(duration, 2),
                            "max_people": crowd_data["max_people"],
                            "location_x": round(
                                crowd_data["location_x"], 2
                            ),
                            "location_y": round(
                                crowd_data["location_y"], 2
                            ),
                            "members": str(
                                sorted(crowd_data["members"])
                            ),
                        })

                    ended_crowds.append(crowd_id)

            for crowd_id in ended_crowds:
                del active_crowds[crowd_id]

            # Annotate and save
            annotated_frame = draw_detections(
                frame.copy(),
                detections,
                current_crowds,
            )

            out.write(annotated_frame)

    finally:
        # Close active crowds at end of video
        final_time = frame_count / fps

        for crowd_id, crowd_data in active_crowds.items():
            start_time = crowd_data["start_time"]
            end_time = min(
                crowd_data["last_seen"],
                final_time,
            )
            duration = end_time - start_time

            if duration >= MIN_CROWD_DURATION:
                crowd_events.append({
                    "crowd_id": crowd_id,
                    "start_time": round(start_time, 2),
                    "end_time": round(end_time, 2),
                    "duration": round(duration, 2),
                    "max_people": crowd_data["max_people"],
                    "location_x": round(
                        crowd_data["location_x"], 2
                    ),
                    "location_y": round(
                        crowd_data["location_y"], 2
                    ),
                    "members": str(
                        sorted(crowd_data["members"])
                    ),
                })

        cap.release()
        out.release()

    return crowd_events


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n==============================================")
    print(" MARKET CROWD MONITORING SYSTEM")
    print(" YOLOv8 + ByteTrack + BFS")
    print("==============================================\n")

    model = load_model()

    crowd_events = process_video(
        model,
        VIDEO_PATH,
        OUTPUT_VIDEO_PATH,
    )

    if crowd_events:
        events_df = pd.DataFrame(crowd_events)

        # Sort chronologically
        events_df = events_df.sort_values(
            by="start_time"
        ).reset_index(drop=True)

        events_df.to_csv(
            OUTPUT_CSV_PATH,
            index=False,
        )

        print("\n==============================================")
        print(" RESULTS")
        print("==============================================")
        print(f"Total crowd events : {len(events_df)}")
        print(f"Output video       : {OUTPUT_VIDEO_PATH}")
        print(f"Output CSV         : {OUTPUT_CSV_PATH}")

        print("\nFirst few events:")
        print(events_df.head(10).to_string(index=False))

    else:
        # Still create an empty CSV with headers
        empty_columns = [
            "crowd_id",
            "start_time",
            "end_time",
            "duration",
            "max_people",
            "location_x",
            "location_y",
            "members",
        ]

        pd.DataFrame(
            columns=empty_columns
        ).to_csv(
            OUTPUT_CSV_PATH,
            index=False,
        )

        print("\nNo significant crowd events detected.")
        print("An empty CSV was created:", OUTPUT_CSV_PATH)


if __name__ == "__main__":
    main()
