from detecto.core import Model
from detecto.visualize import show_labeled_image
from PIL import Image
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
model = Model()

image_path = "testimgs/test26.jpg"
image = Image.open(image_path)
# print(model.predict(image))

labels, boxes, scores = model.predict(image)
print(boxes)
print(scores)
scores = scores.tolist()

free_chairs_count = 0
people_count = []
total_chairs = 0

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

people_list = []
index = 0

for label, score in zip(labels, scores):
    if label == 'person' and score > 0.:
        people_list.append(boxes[index].tolist())
    index=index+1
        
# print(people)
chair_list = []
index = 0

for label, score in zip(labels, scores):
    if label == 'chair' and score > 0.4:
        chair_list.append(boxes[index].tolist())
    index=index+1

fig, ax = plt.subplots()
ax.imshow(image)

# Draw bounding boxes for people
# for person_box in people_list:
#     x_min, y_min, x_max, y_max = person_box
#     rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
#                          fill=False, edgecolor='r', linewidth=2)
#     ax.add_patch(rect)

occ=[]

for chair_box in chair_list:
    x_min, y_min, x_max, y_max = chair_box
    occupied = 0
    for person_box in people_list:
        x_min1, y_min1, x_max1, y_max1 = person_box
        intersection = iou(chair_box, person_box)
        # print(intersection)
        if intersection > 0.15: 
        # and y_min1 < y_max:
            # rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
            #                      fill=False, edgecolor='r', linewidth=2)
            # ax.add_patch(rect)
            occupied = 1
            occ.append(1)
    if not occupied:
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                             fill=False, edgecolor='b', linewidth=2)
        occ.append(0)
        ax.add_patch(rect)
    
print("occupied =",occ.count(1))
print("Total chairs =",len(chair_list))
print("Free chairs =",len(chair_list) - occ.count(1))

info_text = f"Occupied: {occ.count(1)}\nTotal Chairs: {len(chair_list)}\nFree Chairs: {len(chair_list) - occ.count(1)}"
ax.text(0.35, 1.1, info_text, transform=ax.transAxes,
        fontsize=14, color='black', backgroundcolor='white')


x1 = len(chair_list) - occ.count(1)
x2 = len(chair_list)
percentage = x1/x2
print(f"Percentage of free chairs = {percentage*100:.2f}%") #The accuracy we need
plt.show()