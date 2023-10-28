from detecto.core import Model
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
model = Model()

def iou(b1, b2):
    #coordinates of the intersection rectangle
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    
    #area of the rectangle
    area_intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    
    box1_area = (b1[2] - b1[0] + 1) * (b1[3] - b1[1] + 1)
    box2_area = (b2[2] - b2[0] + 1) * (b2[3] - b2[1] + 1)
    
    area_union = box1_area + box2_area - area_intersection
    
    #iou calculation
    intersection = area_intersection / float(area_union)
    return intersection

# Function to draw bounding boxes and display information on frames
def draw_bounding_boxes(frame, labels, boxes, scores):
    people_list = []
    chair_list = []
    occ = []

    for label, score, box in zip(labels, scores, boxes):
        x_min, y_min, x_max, y_max = box
        if label == 'person' and score > 0.5:
            people_list.append(box.tolist())
        elif label == 'chair' and score > 0.4:
            chair_list.append(box.tolist())

    for chair_box in chair_list:
        x_min, y_min, x_max, y_max = chair_box
        occupied = 0
        for person_box in people_list:
            intersection = iou(chair_box, person_box)
            if intersection > 0.15:
                occupied = 1
                occ.append(1)
        if not occupied:
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
            occ.append(0)

    info_text = f"Occupied: {occ.count(1)} Total Chairs: {len(chair_list)} Free Chairs: {len(chair_list) - occ.count(1)}"
    cv2.putText(frame, info_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255,0,0), 5)

    return frame, occ

# Video path
video_path = "images/time_lapse_video_of_people (1080p).mp4"  # Replace this with your video file path

# Open the video file
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB (OpenCV uses BGR by default)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert frame to PIL Image
    image = Image.fromarray(frame_rgb)

    # Get object detection predictions
    labels, boxes, scores = model.predict(image)

    # Draw bounding boxes and display information
    frame_with_boxes, occupancy = draw_bounding_boxes(frame.copy(), labels, boxes, scores)

    # Show the frame with bounding boxes and information
    cv2.imshow("Object Detection", frame_with_boxes)

    # Break the loop if 'q' key is pressed
    key = cv2.waitKey(1)

    if key==27:
        break

cap.release()
cv2.destroyAllWindows()