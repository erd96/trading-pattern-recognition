

import random as rd
import requests
import pandas as pd
import mplfinance as mpf
import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
import datagen_mc_rawdata_rel as datagen
from tensorflow.keras.models import load_model

def generate_bitcoin_chart(num_data_points:int=0) -> (np.array, dict):
    num_data_points = num_data_points if num_data_points else rd.randint(6, 20)

    url = f"https://api.binance.com/api/v3/klines"
    params = {
        "symbol": "BTCUSDT",
        "interval": "1h",
        "limit": num_data_points
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raises an exception if the request is unsuccessful
        response = response.json()
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    print(response)
    # Candlestick data within the boundary lines
    dates = [pd.to_datetime(entry[0], unit='ms') for entry in response]
    open_data = [float(entry[1]) for entry in response]
    high_data = [float(entry[2]) for entry in response]
    low_data = [float(entry[3]) for entry in response]
    close_data = [float(entry[4]) for entry in response]
    volume = [float(entry[5]) for entry in response]
    
    img = datagen.scale_and_create_matrix(open_data, high_data, low_data, close_data, volume)
    
    data_dict = {i: {"date": d, "open": o, "high": h, "low": l, "close": c} for i, (d, o, h, l, c) in enumerate(zip(dates, open_data, high_data, low_data, close_data))}

    return img, data_dict
    

img, data = generate_bitcoin_chart(100)


def calculate_iou(box1, box2):
    # Calculate the Intersection over Union (IoU) of two bounding boxes
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x_intersection = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_intersection = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    intersection = x_intersection * y_intersection

    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection

    iou = intersection / union if union > 0 else 0
    return iou


pattern_classes = ["Rising Wedge", "Falling Wedge", "Symmetrical Triangle", "Ascending Triangle", "Descending Triangle"]

def remove_duplicate_bboxes(bboxes):
    
    bboxes = [bbox for bbox in bboxes if bbox[2] >= 6]
    # Remove duplicate bounding boxes based on IoU threshold
    iou_threshold = 0.5
    unique_bboxes = []

    for bbox in bboxes:
        is_duplicate = False

        for unique_bbox in unique_bboxes:
            iou = calculate_iou(bbox, unique_bbox)

            if iou > iou_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_bboxes.append(bbox)

    
    return unique_bboxes


print(img.shape)
cv2.imshow("Bitcoin Chart", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

def selectiveSearchFast(img): # OpenCV's Selective search fast algorithm. Returns arrays of type [x1,y1, w, h]. 
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    # ss.switchToSelectiveSearchQuality()
    bounding_boxes = ss.process()
    
    return bounding_boxes


# Convert the grayscale image to RGB for selective search 
img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

# Run selective search on the RGB image
bboxes = selectiveSearchFast(img_rgb)

bboxes = remove_duplicate_bboxes(bboxes)

print(len(bboxes))
print("Input Image Dimensions:", img.shape)
print("Generated Bounding Boxes:", bboxes)


# Show the bounding boxes 
while True:
    imOut = img.copy()
    for i, rect in enumerate(bboxes):
        x, y, w, h = rect
        cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
    
    cv2.imshow("Output", imOut)
    
    k = cv2.waitKey(0) & 0xFF
    
    if k == 113:  # q to quit
        break

cv2.destroyAllWindows()


def create_bounding_box_images(image, bounding_boxes):
    bounding_box_images = []

    for i, bbox in enumerate(bounding_boxes):
        x, y, w, h = bbox
        print(f"Bounding box {i}: x={x}, y={y}, w={w}, h={h}")
        top = y
        bottom = y + h - 1
        
        # Check if the top or bottom rows are already white, reduce dimensions if necessary
        while top < bottom and np.all(image[top, x:x + w] == 255):
            top += 1

        while top < bottom and np.all(image[bottom, x:x + w] == 255):
            bottom -= 1
            
        # Check top row
        while top > 0 and not np.all(image[top-1, x:x + w] == 255):
            top -= 1

        # Check bottom row
        while bottom < image.shape[0] - 1 and not np.all(image[bottom+1, x:x + w] == 255):
            bottom += 1

        # Update the bounding box
        new_y = top
        new_h = bottom - top + 1  # Adjust height based on top and bottom

        bbox_image = image[new_y:new_y + new_h, x:x + w]
        bounding_box_images.append(((x, new_y, w, new_h), datagen.pad_image(bbox_image)))

    return bounding_box_images

bounding_box_images_array = create_bounding_box_images(img, bboxes)

model_path = 'patterns_classification_model_rawdata_rel.h5'
model = load_model(model_path)

results = []

print(bounding_box_images_array[0][1].shape)
cv2.imshow("Bounding Box Image", bounding_box_images_array[1][1])
cv2.waitKey(0)
cv2.destroyAllWindows()

for i, image in enumerate(bounding_box_images_array):
    bbox, img = image
    grayscale_image = img  # Assuming img is already grayscale
    resized_image = cv2.resize(grayscale_image, (32, 32), interpolation=cv2.INTER_NEAREST)
    reshaped_image = np.reshape(resized_image, (32, 32, 1))/255
    raw_prediction = model.predict(np.array([reshaped_image]))[0]
    # prediction_label = pattern_classes[np.argmax(raw_prediction)]
    # results.append((bbox, raw_prediction, prediction_label))
    if np.max(raw_prediction) > 0.97: 
        results.append((bbox, raw_prediction, pattern_classes[np.argmax(raw_prediction)]))



identified_patterns = []

def non_max_suppression(boxes):
    if len(boxes) == 0:
        return []

    # Sort the boxes based on the maximum confidence scores
    boxes = sorted(boxes, key=lambda x: np.max(x[1]), reverse=True)

    # Initialize a list to store the selected bounding boxes after NMS
    selected_boxes = [boxes[0]]

    # Loop over the remaining boxes
    for box in boxes[1:]:
        # Check if the current box overlaps with any of the selected boxes with the same prediction
        overlap = False
        for selected_box in selected_boxes:
            if box[2] == selected_box[2]:  # Check if predictions are the same
                x1 = max(box[0][0], selected_box[0][0])
                y1 = max(box[0][1], selected_box[0][1])
                x2 = min(box[0][0] + box[0][2], selected_box[0][0] + selected_box[0][2])
                y2 = min(box[0][1] + box[0][3], selected_box[0][1] + selected_box[0][3])

                intersection = max(0, x2 - x1) * max(0, y2 - y1)
                area_box = box[0][2] * box[0][3]
                area_selected = selected_box[0][2] * selected_box[0][3]
                union = area_box + area_selected - intersection

                # Calculate the IoU
                iou = intersection / union

                # If the boxes overlap, set the flag to True
                if iou > 0:
                    overlap = True
                    break

        # If the box doesn't overlap with any selected box with the same prediction, add it
        if not overlap:
            selected_boxes.append(box)

    return selected_boxes


filtered_results = non_max_suppression(results)

# Start index, stop index, classification
identified_patterns = sorted([(b[0][0], b[0][2], np.max(b[1]), b[2]) for b in filtered_results], key=lambda x: x[0] )
print(identified_patterns)
print("BBoxes # After NMS: ", len(filtered_results))



# # Define colors for each BGR format
class_colors = {
    "Rising Wedge": "green",            
    "Falling Wedge": "darkorange",            
    "Symmetrical Triangle": "blue",     
    "Ascending Triangle": "purple",     
    "Descending Triangle": "magenta"    
}

mco = [None for i in range(len(data))]
for box in identified_patterns:
    for i in range(box[0], box[0] + box[1], 1):
        mco[i] = class_colors[box[3]]

# Convert data_dict to a list of dictionaries
ohlc_data = list(data.values())


# Convert data to a DataFrame
df = pd.DataFrame(ohlc_data)
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)

# Plot using mplfinance
mpf.plot(df, type='candle', style='yahoo', volume=False, marketcolor_overrides=mco)
# mpf.plot(df, type='candle', style='yahoo', volume=False)


