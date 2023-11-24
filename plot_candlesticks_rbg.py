

import random as rd
import requests
import pandas as pd
import mplfinance as mpf
import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import ttk
from mplfinance.original_flavor import candlestick_ohlc
from matplotlib.dates import date2num



def generate_bitcoin_chart(num_data_points=0):
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

    # Candlestick data within the boundary lines
    dates = [pd.to_datetime(entry[0], unit='ms') for entry in response]
    open_data = [float(entry[1]) for entry in response]
    high_data = [float(entry[2]) for entry in response]
    low_data = [float(entry[3]) for entry in response]
    close_data = [float(entry[4]) for entry in response]
    volume = [float(entry[5]) for entry in response]
    
    df= pd.DataFrame({'date':dates, 'open':open_data, "high":high_data, "low":low_data, "close":close_data})
    
    # Convert date to numerical format
    df['numdate'] = date2num(df['date'])
    
    x_min, x_max = df['numdate'].min(), df['numdate'].max()
    num_data_points = len(df['numdate'])
    width_factor = 0.8  # Adjust this factor based on your preference
  # Calculate dynamic width
    width = (x_max - x_min) / num_data_points 
  # Create candlestick chart with dynamic width
    fig, ax = plt.subplots()
    candlestick_ohlc(ax, zip(df['numdate'], df['open'], df['high'], df['low'], df['close']),
                 width=width, colorup='green', colordown='red')


    ax.axis('off')
    # fig.set_size_inches(5, 5)  # Set width and height to the same value (e.g., 5 inches)
    ax.set_aspect('auto')
        # Set y-axis limits manually to ensure accurate representation of candlestick heights
    ax.set_ylim(df['low'].min(), df['high'].max())
    
    # Get the axes object
    axes = plt.gca()
    # Apply tight layout
    plt.tight_layout()
    # Get the x-pixel coordinates of the left and right sides of the first candlestick
    x_data = df['numdate'].iloc[0]  # Use the first data point for simplicity
    x_pixel_left = ax.transData.transform((x_data - width / 2, 0))[0]
    x_pixel_right = ax.transData.transform((x_data + width / 2, 0))[0]

    # Calculate the candlestick width in pixels
    candlestick_width_pixels = x_pixel_right - x_pixel_left

    
    save_path = "bitcoin_chart.jpg"
    fig.savefig(save_path, format='jpg')  # Save the image to the specified path
    plt.close(fig)
    return save_path, df, candlestick_width_pixels



# def generate_bitcoin_chart(num_data_points=0):
#     num_data_points = num_data_points if num_data_points else rd.randint(6, 20)

#     url = f"https://api.binance.com/api/v3/klines"
#     params = {
#         "symbol": "BTCUSDT",
#         "interval": "1h",
#         "limit": num_data_points
#     }

#     try:
#         response = requests.get(url, params=params)
#         response.raise_for_status()  # Raises an exception if the request is unsuccessful
#         response = response.json()
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return None

#     # Candlestick data within the boundary lines
#     dates = [pd.to_datetime(entry[0], unit='ms') for entry in response]
#     open_data = [float(entry[1]) for entry in response]
#     high_data = [float(entry[2]) for entry in response]
#     low_data = [float(entry[3]) for entry in response]
#     close_data = [float(entry[4]) for entry in response]
#     volume = [float(entry[5]) for entry in response]
#     print("Vol: ", volume)


#     fig, ax = plt.subplots()
#     ax.bar(range(len(volume)), volume, color='black', width=0.8)

#     # Remove spines (borders) from the plot
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     ax.spines['left'].set_visible(False)

#     plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
#     plt.show()

#     df = pd.DataFrame({'Date': dates, 'Open': open_data, 'High': high_data, 'Low': low_data, 'Close': close_data})

#     df.set_index('Date', inplace=True)
#     custom_style = mpf.make_mpf_style(base_mpf_style='charles', gridstyle='')
#     figsize_multiplier = 0.5
#     fig, axlist = mpf.plot(df, type='candle', style=custom_style, title='', ylabel='Price', axisoff=True,   scale_padding=0, returnfig=True, closefig=True)

#     img_bytes_io = io.BytesIO()
#     fig.savefig(img_bytes_io, format='jpg')
#     img_bytes_io.seek(0)

#     return grayscale_and_normalize(img_bytes_io)


def preprocess_image(img):
    img.seek(0)
    img_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
    img_cv2 = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    img_gray_resized = cv2.resize(img_gray, (128, 128), interpolation=cv2.INTER_AREA)
    img_gray_resized = img_gray_resized.reshape((*img_gray_resized.shape, 1))
    img_gray_resized = img_gray_resized.astype('float32') / 255
    return img_gray_resized


def grayscale_and_normalize(img):
    img.seek(0)
    img_bytes = np.asarray(bytearray(img.read()))
    img_cv2 = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_cv2 = img_cv2.astype('float32') / 255
    return img_cv2


def rgb_and_normalize(img_path):
    img_cv2 = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)  # Swap BGR to RGB
    img_cv2 = img_cv2.astype('float32') / 255
    return img_cv2

chart_path, df, cwidth = generate_bitcoin_chart(50)


pattern_classes = ["Rising Wedge", "Falling Wedge", "Symmetrical Triangle", "Ascending Triangle", "Descending Triangle" ,"No Patterns" ]

# Load and display the saved chart
img_array = rgb_and_normalize(chart_path)
plt.imshow(img_array)
plt.axis('off')
plt.show()

# plt.imshow(chart_image)
# plt.axis('off')
# plt.show()

def selectiveSearchFast(img): # OpenCV's Selective search fast algorithm. Returns arrays of type [x1,y1, w, h]. 
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    # ss.switchToSelectiveSearchFast()
    ss.switchToSelectiveSearchQuality()
    bounding_boxes = ss.process()
    return bounding_boxes

bboxes = selectiveSearchFast(img_array)
# bboxes = selectiveSearchFast(chart_image)
bboxes = [bbox for bbox in bboxes if bbox[2] >= 6 * cwidth]

while True:
    imOut = img_array.copy()
    for i, rect in enumerate(bboxes):
        x, y, w, h = rect
        cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
    
    cv2.imshow("Output", cv2.cvtColor(imOut, cv2.COLOR_RGB2BGR))
    
    k = cv2.waitKey(0) & 0xFF
    
    if k == 113:  # q to quit
        break

cv2.destroyAllWindows()

# while True:
#     imOut = chart_image.copy()
#     for i, rect in enumerate(bboxes):
#         x, y, w, h = rect
#         cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
    
#     cv2.imshow("Output", imOut)
    
#     k = cv2.waitKey(0) & 0xFF
    
#     if k == 113:  # q to quit
#         break

# cv2.destroyAllWindows()


def create_bounding_box_images(image, bounding_boxes):
    bounding_box_images = []

    for i, bbox in enumerate(bounding_boxes):
        x, y, w, h = bbox
        print(f"Bounding box {i}: x={x}, y={y}, w={w}, h={h}")
        top = y
        bottom = y + h - 1
        
        # Check if the top or bottom rows are already white, reduce dimensions if necessary
        while top < bottom and np.all(image[top, x:x + w] == 1):
            top += 1

        while top < bottom and np.all(image[bottom, x:x + w] == 1):
            bottom -= 1
            
        # Check top row
        while top > 0 and not np.all(image[top-1, x:x + w] == 1):
            top -= 1

        # Check bottom row
        while bottom < image.shape[0] - 1 and not np.all(image[bottom+1, x:x + w] == 1):
            bottom += 1

        # Update the bounding box
        new_y = top
        new_h = bottom - top + 1  # Adjust height based on top and bottom

        bbox_image = image[new_y:new_y + new_h, x:x + w]
        bounding_box_images.append(((x, new_y, w, new_h), bbox_image))

    return bounding_box_images

bounding_box_images_array = create_bounding_box_images(img_array, bboxes)
# bounding_box_images_array = create_bounding_box_images(chart_image, bboxes)

model_path = 'patterns_classification_model_5.h5'
model = load_model(model_path)

results = []
print("BBoxes # Original: ", len(results))

resized_width, resized_height = 128, 128

for i, image in enumerate(bounding_box_images_array):
    bbox, img = image
    resized_image = cv2.resize(img, (resized_width, resized_height))
    reshaped_image = np.reshape(resized_image, (1, resized_width, resized_height, 3))  # Add batch dimension
    raw_prediction = model.predict(reshaped_image)
    prediction_label = "No Patterns" if np.max(raw_prediction) < 0.99 else pattern_classes[np.argmax(raw_prediction)]
    # prediction_label = pattern_classes[np.argmax(raw_prediction)]
    results.append((bbox, raw_prediction, prediction_label))
    


results = [(bbox, raw_prediction, prediction) for bbox, raw_prediction, prediction in results if prediction != "No Patterns"]

print("BBoxes # Filtered: ", len(results))



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

print("BBoxes # After NMS: ", len(filtered_results))


# Define colors for each BGR format
class_colors = {
    "Rising Wedge": (0, 255, 0),            # Green
    "Falling Wedge": (0, 0, 255),            # Red
    "Symmetrical Triangle": (255, 0, 0),     # Blue
    "Ascending Triangle": (0, 255, 255),     # Yellow
    "Descending Triangle": (255, 0, 255),    # Magenta
    'No Patterns': (128, 128, 128)           # Gray
}

# # Print raw prediction and prediction for each final bounding box
# for i, rect in enumerate(filtered_results):
#     bbox, raw_prediction, prediction = rect
#     print(f"BBox {i + 1} : {bbox} - Raw Prediction: {max(raw_prediction)}, Predicted Class: {prediction}")
    
# while True:
#     imOut = chart_image.copy()
#     for i, rect in enumerate(filtered_results):
#         x, y, w, h = rect[0]
#         prediction_label = rect[2]
#         color = class_colors.get(prediction_label, (0, 0, 0))
#         cv2.rectangle(imOut, (x, y), (x+w, y+h), color, 1)

#     cv2.imshow("Output", imOut)

#     k = cv2.waitKey(0) & 0xFF

#     if k == 113:  # q to quit
#         break

# cv2.destroyAllWindows()

def toggle_boxes(class_name):
    # imOut = chart_image.copy()
    imOut = img_array.copy()
    for i, rect in enumerate(filtered_results):
        x, y, w, h = rect[0]
        prediction_label = rect[2]
        color = class_colors.get(prediction_label, (0, 0, 0))
        
        if prediction_label == class_name:
            cv2.rectangle(imOut, (x, y), (x+w, y+h), color, 1)

    cv2.imshow("Output", cv2.cvtColor(imOut, cv2.COLOR_RGB2BGR))

# Create a Tkinter window
root = tk.Tk()
root.title("Interactive Chart")

# Create a dropdown for selecting classes
class_names = list(class_colors.keys())
selected_class = tk.StringVar()
class_dropdown = ttk.Combobox(root, textvariable=selected_class, values=class_names)
class_dropdown.set(class_names[0])
class_dropdown.pack(pady=10)

# Create a button to toggle bounding boxes
toggle_button = ttk.Button(root, text="Toggle Bounding Boxes", command=lambda: toggle_boxes(selected_class.get()))
toggle_button.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()

# Cleanup after the Tkinter loop ends
cv2.destroyAllWindows()