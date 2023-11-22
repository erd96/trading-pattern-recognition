from numpy.random.mtrand import randint
import numpy as np
import random as rd
import cv2
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
from datetime import timedelta

def min_max_normalize(data:list, feature_range:tuple=(0,1)) -> list:
    scaler = MinMaxScaler(feature_range=feature_range)
    data = np.array(data).reshape(-1,1)
    normalized = scaler.fit_transform(data)
    
    return normalized.flatten() if feature_range == (0,1) else normalized.flatten().astype(int)

def generate_volume(num_data_points, pattern_type):
    volume_data = []

    for i in range(num_data_points):
        if pattern_type == "rising_wedge":
            if i == num_data_points - 1:
                volume = np.random.uniform(80, 120)
            elif i == num_data_points // 2:
                volume = np.random.uniform(20, 40)
            else:
                volume = np.random.uniform(10, 30)
        elif pattern_type == "falling_wedge":
            if i == num_data_points - 1:
                volume = np.random.uniform(80, 120)
            elif i == num_data_points // 2:
                volume = np.random.uniform(20, 40)
            else:
                volume = np.random.uniform(10, 30)
        elif pattern_type == "symmetrical_triangle":
            if i == num_data_points // 2:
                volume = np.random.uniform(80, 120)
            else:
                volume = np.random.uniform(10, 30)
        elif pattern_type == "ascending_triangle":
            if i == num_data_points - 1:
                volume = np.random.uniform(80, 120)
            elif i == num_data_points // 2:
                volume = np.random.uniform(20, 40)
            else:
                volume = np.random.uniform(10, 30)
        elif pattern_type == "descending_triangle":
            if i == num_data_points - 1:
                volume = np.random.uniform(80, 120)
            elif i == num_data_points // 2:
                volume = np.random.uniform(20, 40)
            else:
                volume = np.random.uniform(10, 30)
        else:
            volume = np.random.uniform(10, 30)

        volume_data.append(volume)
        
    return volume_data

def resize_images(image_list, target_size=(128, 128)):
    # Resize each image in the list using cv2
    resized_images = [cv2.resize(img, target_size, interpolation=cv2.INTER_NEAREST) for img in image_list]
    
    # Add channel dimension
    resized_images = np.array(resized_images)
    resized_images = np.expand_dims(resized_images, axis=-1)
    
    return resized_images


def pad_image(data_arr:np.array) -> np.array:
     # Calculate the number of columns to pad on each side
    num_cols_to_pad = data_arr.shape[0] - data_arr.shape[1]

    left_pad = num_cols_to_pad // 2
    right_pad = num_cols_to_pad - left_pad

    # Pad the array with equal padding on both sides
    padded_data_arr = np.pad(data_arr, ((0, 0), (left_pad, right_pad)), mode='constant', constant_values=255)
    return padded_data_arr.astype(np.uint8)

def scale_and_create_matrix(o, h, l, c, v):
    """
    Scale each category (O, H, L, C) independently and create a candlestick matrix.

    Parameters:
    - open_data: numpy array, open prices
    - high_data: numpy array, high prices
    - low_data: numpy array, low prices
    - close_data: numpy array, close prices

    Returns:
    - candlestick_matrix: numpy array, candlestick matrix for CNN
    """
    
    # Ensure each input is a NumPy array
    combined_data = np.array(o + h + l + c)
    combined_data = combined_data.reshape(-1, 1)
    
    # Apply Min-Max scaling to all values combined
    scaler = MinMaxScaler(feature_range=(0, len(o)*4))
    scaled_data = scaler.fit_transform(combined_data)
    # Round the scaled values to integers
    scaled_data_rounded = np.round(scaled_data)
    scaled_data_rounded = scaled_data_rounded.flatten()
    
    # Separate the rounded values back into their respective arrays
    scaled_o = scaled_data_rounded[:len(o)]
    scaled_h = scaled_data_rounded[len(o):2*len(h)]
    scaled_l = scaled_data_rounded[2*len(l):3*len(l)]
    scaled_c = scaled_data_rounded[3*len(c):]

    candlesticks = np.array([np.array([o,h,l,c]) for o,h,l,c in zip(scaled_o, scaled_h, scaled_l, scaled_c)])
    arrmax = np.max(candlesticks)
    temp = []
    for candlestick in candlesticks:
        arr = np.ones(int(arrmax))*255
        # print(arr.shape)
        for i in range(int(candlestick[2]), int(candlestick[1]),1):
            # print(i)
            arr[i] = 0
        
        maxv,minv = max(int(candlestick[0]), int(candlestick[3])), min(int(candlestick[0]), int(candlestick[3]))
        for j in range(minv, maxv,1):
            # print(j)
            arr[j] = 32
        # print("\n")
        temp.append(arr)
            
    data_arr = np.array(temp).T
    
    flipped_data_arr = np.flipud(data_arr)
    
    return  flipped_data_arr.astype(np.uint8)
    # return  data_arr.astype(np.uint8)


def generate_rising_wedge(index=0):
    num_data_points = rd.randint(6, 30)
    # Generate random slopes for the lines
    lower_slope = np.random.uniform(0.1, 2)
    upper_slope = np.random.uniform(0.1, lower_slope - 0.2)  # Ensuring upper slope is less than lower slope
    # Generate random bias for the upper line
    bias = np.random.uniform(0.1, 0.5)

    # Generate x values
    x = np.linspace(0, num_data_points, num_data_points)

    # y values for each boundary line
    lower_line = lower_slope * x
    upper_line = upper_slope * (x - num_data_points) + lower_line[-1] + bias

    # Candlestick data within the boundary lines
    open_data = []
    close_data = []
    high_data = []
    low_data = []
    volume = generate_volume(num_data_points, "rising_wedge")
    split =  num_data_points//3
    start, mid, end = x[0:split], x[split :split*2], x[split*2:]
    ltouches = np.concatenate([np.where(x == rd.choice(start))[0], np.where(x == rd.choice(mid))[0], np.where(x == rd.choice(end))[0]]).tolist()
    utouches = np.concatenate([np.where(x == rd.choice(start))[0], np.where(x == rd.choice(mid))[0], np.where(x == rd.choice(end))[0]]).tolist()

    current_price = (upper_line[0] + lower_line[0]) / 2
    for i in range(num_data_points):
        if i == 0:
            open_price = current_price
        else:
            open_price = close_data[i - 1]

        if i in ltouches:
          close_or_low = rd.choice([True, False])
          if close_or_low:
            close_price =  lower_line[i]  
            low_price =  np.random.uniform(lower_line[i], upper_line[i])
          else: 
            close_price =  np.random.uniform(lower_line[i], upper_line[i])  
            low_price =  lower_line[i]
          high_price =  np.random.uniform(lower_line[i], upper_line[i])
          
        elif i in utouches:
          close_or_high = rd.choice([True, False])
          if close_or_high:
            close_price =  upper_line[i]  
            high_price =  np.random.uniform(lower_line[i], upper_line[i])
          else: 
            close_price =  np.random.uniform(lower_line[i], upper_line[i])  
            high_price =  upper_line[i]
          low_price =  np.random.uniform(lower_line[i], upper_line[i])
        else:
          close_price =  np.random.uniform(lower_line[i], upper_line[i])  
          high_price =  np.random.uniform(lower_line[i], upper_line[i])
          low_price =  np.random.uniform(lower_line[i], upper_line[i])

        if close_price > open_price : 
            high_price = high_price if high_price > close_price else close_price
            low_price = low_price if low_price < open_price else open_price
        if close_price < open_price : 
            high_price = high_price if high_price > open_price else open_price
            low_price = low_price if low_price < close_price else close_price
            
        open_data.append(open_price)
        close_data.append(close_price)
        high_data.append(high_price)
        low_data.append(low_price)

        current_price = close_price  # Set the current price for the next iteration
    
    img = scale_and_create_matrix(open_data, high_data, low_data, close_data, volume)
    label = np.array([1,0,0,0,0])
    return img,label


def generate_falling_wedge(index=0):
    num_data_points = rd.randint(6, 30)
    # Generate random slopes for the lines
    upper_slope = -np.random.uniform(0.1, 2)
    lower_slope = np.random.uniform(-0.1, upper_slope +0.2)  # Ensuring upper slope is less than lower slope
    # Generate random bias for the upper line
    bias = -np.random.uniform(0.1, 0.5)

    # Generate x values
    x = np.linspace(0, num_data_points, num_data_points)

    # y values for each boundary line
    upper_line = upper_slope * x
    lower_line = lower_slope * (x - num_data_points) + upper_line[-1] + bias

    # Candlestick data within the boundary lines
    open_data = []
    close_data = []
    high_data = []
    low_data = []
    volume = generate_volume(num_data_points, "falling_wedge")
    split =  num_data_points//3
    start, mid, end = x[0:split], x[split :split*2], x[split*2:]
    ltouches = np.concatenate([np.where(x == rd.choice(start))[0], np.where(x == rd.choice(mid))[0], np.where(x == rd.choice(end))[0]]).tolist()
    utouches = np.concatenate([np.where(x == rd.choice(start))[0], np.where(x == rd.choice(mid))[0], np.where(x == rd.choice(end))[0]]).tolist()

    current_price = (upper_line[0] + lower_line[0]) / 2
    for i in range(num_data_points):
        if i == 0:
            open_price = current_price
        else:
            open_price = close_data[i - 1]

        if i in ltouches:
          close_or_low = rd.choice([True, False])
          if close_or_low:
            close_price =  lower_line[i]  
            low_price =  np.random.uniform(lower_line[i], upper_line[i])
          else: 
            close_price =  np.random.uniform(lower_line[i], upper_line[i])  
            low_price =  lower_line[i]
          high_price =  np.random.uniform(lower_line[i], upper_line[i])
          
        elif i in utouches:
          close_or_high = rd.choice([True, False])
          if close_or_high:
            close_price =  upper_line[i]  
            high_price =  np.random.uniform(lower_line[i], upper_line[i])
          else: 
            close_price =  np.random.uniform(lower_line[i], upper_line[i])  
            high_price =  upper_line[i]
          low_price =  np.random.uniform(lower_line[i], upper_line[i])
        else:
          close_price =  np.random.uniform(lower_line[i], upper_line[i])  
          high_price =  np.random.uniform(lower_line[i], upper_line[i])
          low_price =  np.random.uniform(lower_line[i], upper_line[i])

        if close_price > open_price : 
            high_price = high_price if high_price > close_price else close_price
            low_price = low_price if low_price < open_price else open_price
        if close_price < open_price : 
            high_price = high_price if high_price > open_price else open_price
            low_price = low_price if low_price < close_price else close_price
            
        open_data.append(open_price)
        close_data.append(close_price)
        high_data.append(high_price)
        low_data.append(low_price)

        current_price = close_price  # Set the current price for the next iteration

    img = scale_and_create_matrix(open_data, high_data, low_data, close_data, volume)
    label = np.array([0,1,0,0,0])
        
    return img,label




def generate_symmetrical_triangle(index=0):
    num_data_points = rd.randint(6, 30)
    # Generate random slopes for the lines
    upper_slope = -np.random.uniform(0.1, 2)
    lower_slope = -upper_slope  # Ensuring upper slope is less than lower slope
    # Generate random bias for the upper line
    bias = -np.random.uniform(0.1, 0.5)

    # Generate x values
    x = np.linspace(0, num_data_points, num_data_points)

    # y values for each boundary line
    upper_line = upper_slope * x
    lower_line = lower_slope * (x - num_data_points) + upper_line[-1] + bias

    # Candlestick data within the boundary lines
    open_data = []
    close_data = []
    high_data = []
    low_data = []
    volume = generate_volume(num_data_points, "symmetrical_triangle")
    split =  num_data_points//3
    start, mid, end = x[0:split], x[split :split*2], x[split*2:]
    ltouches = np.concatenate([np.where(x == rd.choice(start))[0], np.where(x == rd.choice(mid))[0], np.where(x == rd.choice(end))[0]]).tolist()
    utouches = np.concatenate([np.where(x == rd.choice(start))[0], np.where(x == rd.choice(mid))[0], np.where(x == rd.choice(end))[0]]).tolist()

    current_price = (upper_line[0] + lower_line[0]) / 2
    for i in range(num_data_points):
        if i == 0:
            open_price = current_price
        else:
            open_price = close_data[i - 1]

        if i in ltouches:
          close_or_low = rd.choice([True, False])
          if close_or_low:
            close_price =  lower_line[i]  
            low_price =  np.random.uniform(lower_line[i], upper_line[i])
          else: 
            close_price =  np.random.uniform(lower_line[i], upper_line[i])  
            low_price =  lower_line[i]
          high_price =  np.random.uniform(lower_line[i], upper_line[i])
          
        elif i in utouches:
          close_or_high = rd.choice([True, False])
          if close_or_high:
            close_price =  upper_line[i]  
            high_price =  np.random.uniform(lower_line[i], upper_line[i])
          else: 
            close_price =  np.random.uniform(lower_line[i], upper_line[i])  
            high_price =  upper_line[i]
          low_price =  np.random.uniform(lower_line[i], upper_line[i])
        else:
          close_price =  np.random.uniform(lower_line[i], upper_line[i])  
          high_price =  np.random.uniform(lower_line[i], upper_line[i])
          low_price =  np.random.uniform(lower_line[i], upper_line[i])

        if close_price > open_price : 
            high_price = high_price if high_price > close_price else close_price
            low_price = low_price if low_price < open_price else open_price
        if close_price < open_price : 
            high_price = high_price if high_price > open_price else open_price
            low_price = low_price if low_price < close_price else close_price
            
        open_data.append(open_price)
        close_data.append(close_price)
        high_data.append(high_price)
        low_data.append(low_price)

        current_price = close_price  # Set the current price for the next iteration
        
    img = scale_and_create_matrix(open_data, high_data, low_data, close_data, volume)
    label = np.array([0,0,1,0,0])
    return img,label
  
def generate_ascending_triangle(index=0):
    num_data_points = rd.randint(6, 30)

    # Generate x values
    x = np.linspace(0, num_data_points, num_data_points)


    # Generate random slope for the ascending support line
    support_slope = np.random.uniform(0.1, 2)

    # Generate y values for the ascending support line
    support_line = support_slope * x

    # Straight resistance line
    resistance_line = np.ones_like(x) * np.max(support_line) + 0.01
    

    # Candlestick data within the boundary lines
    open_data = []
    close_data = []
    high_data = []
    low_data = []

    split = num_data_points // 3
    start, mid, end = x[0:split], x[split :split*2], x[split*2:]
    ltouches = np.concatenate([np.where(x == rd.choice(start))[0], np.where(x == rd.choice(mid))[0], np.where(x == rd.choice(end))[0]]).tolist()
    utouches = np.concatenate([np.where(x == rd.choice(start))[0], np.where(x == rd.choice(mid))[0], np.where(x == rd.choice(end))[0]]).tolist()
    volume = generate_volume(num_data_points, "ascending_triangle")
    current_price = (support_line[0] + resistance_line[0]) / 2

    for i in range(num_data_points):
        if i == 0:
            open_price = current_price
        else:
            open_price = close_data[i - 1]

        if i in ltouches:
            close_or_low = rd.choice([True, False])
            if close_or_low:
                close_price =  support_line[i]  
                low_price =  np.random.uniform(support_line[i], resistance_line[i])
            else: 
                close_price =  np.random.uniform(support_line[i], resistance_line[i])
                low_price =  support_line[i]
            high_price =  np.random.uniform(support_line[i], resistance_line[i])
        
        elif i in utouches:
            close_or_high = rd.choice([True, False])
            if close_or_high:
                close_price =  resistance_line[i]  
                high_price =  np.random.uniform(support_line[i], resistance_line[i])
            else: 
                close_price =  np.random.uniform(support_line[i], resistance_line[i]) 
                high_price =  resistance_line[i]
            low_price =  np.random.uniform(support_line[i], resistance_line[i])
        else:
            close_price =  np.random.uniform(support_line[i], resistance_line[i])
            high_price =  np.random.uniform(support_line[i], resistance_line[i])
            low_price =  np.random.uniform(support_line[i], resistance_line[i])

        if close_price > open_price : 
            high_price = high_price if high_price > close_price else close_price
            low_price = low_price if low_price < open_price else open_price
        if close_price < open_price : 
            high_price = high_price if high_price > open_price else open_price
            low_price = low_price if low_price < close_price else close_price
            
        open_data.append(open_price)
        close_data.append(close_price)
        high_data.append(high_price)
        low_data.append(low_price)

        current_price = close_price  # Set the current price for the next iteration
        
    img = scale_and_create_matrix(open_data, high_data, low_data, close_data, volume)
    label = np.array([0,0,0,1,0])
    return img,label

  
def generate_descending_triangle(index=0):
    num_data_points = rd.randint(6, 30)

    # Generate x values
    x = np.linspace(0, num_data_points, num_data_points)
    # Generate random slope for the descending resistance line
    resistance_slope = -np.random.uniform(0.1, 2)
    # Generate y values for the descending resistance line
    resistance_line = (resistance_slope * x) + (np.absolute(np.max(x)*resistance_slope)*2)

    # Straight support line
    support_line = np.ones_like(x) * np.min(resistance_line) - 0.01

    # Candlestick data within the boundary lines
    open_data = []
    close_data = []
    high_data = []
    low_data = []
    volume = generate_volume(num_data_points, "descending_triangle")
    split = num_data_points // 3
    start, mid, end = x[0:split], x[split:split*2], x[split*2:]
    ltouches = np.concatenate([np.where(x == np.random.choice(start))[0], np.where(x == np.random.choice(mid))[0], np.where(x == np.random.choice(end))[0]]).tolist()
    utouches = np.concatenate([np.where(x == np.random.choice(start))[0], np.where(x == np.random.choice(mid))[0], np.where(x == np.random.choice(end))[0]]).tolist()

    current_price = (support_line[0] + resistance_line[0]) / 2

    for i in range(num_data_points):
        if i == 0:
            open_price = current_price
        else:
            open_price = close_data[i - 1]

        if i in ltouches:
            close_or_low = np.random.choice([True, False])
            if close_or_low:
                close_price = support_line[i]
                low_price = np.random.uniform(support_line[i], resistance_line[i])
            else:
                close_price = np.random.uniform(support_line[i], resistance_line[i])
                low_price = support_line[i]
            high_price = np.random.uniform(support_line[i], resistance_line[i])
        elif i in utouches:
            close_or_high = np.random.choice([True, False])
            if close_or_high:
                close_price = resistance_line[i]
                high_price = np.random.uniform(support_line[i], resistance_line[i])
            else:
                close_price = np.random.uniform(support_line[i], resistance_line[i])
                high_price = resistance_line[i]
            low_price = np.random.uniform(support_line[i], resistance_line[i])
        else:
            close_price = np.random.uniform(support_line[i], resistance_line[i])
            low_price = np.random.uniform(support_line[i], resistance_line[i])
            high_price = np.random.uniform(support_line[i], resistance_line[i])


        if close_price > open_price : 
            high_price = high_price if high_price > close_price else close_price
            low_price = low_price if low_price < open_price else open_price
        if close_price < open_price : 
            high_price = high_price if high_price > open_price else open_price
            low_price = low_price if low_price < close_price else close_price
            
        open_data.append(open_price)
        close_data.append(close_price)
        high_data.append(high_price)
        low_data.append(low_price)

        current_price = close_price  # Set the current price for the next iteration
    

    img = scale_and_create_matrix(open_data, high_data, low_data, close_data, volume)
    label = np.array([0,0,0,0,1])
    return img,label
  

def generate_no_pattern(index=0):
    choice = np.random.choice([True, False])
    if choice:
        num_data_points = np.random.randint(1, 6)
    else:
        num_data_points = np.random.randint(6, 30)
        
    # Generate random prices for the candlestick data
    open_data = np.random.randint(1, 10, num_data_points)
    close_data = np.random.randint(1, 10, num_data_points)
    high_data = np.random.randint(1, 10, num_data_points)
    low_data = np.random.randint(1, 10, num_data_points)

    # Reshape and then flatten after scaling
    mul = num_data_points // 4 
    img = scale_and_create_matrix(open_data, high_data, low_data, close_data,mul)
    label = [0,0,0,0,0,1]

    # return img,label


classes = ["Rising Wedge", "Falling Wedge", "Symmetrical Triangle", "Ascending Triangle", "Descending Triangle"]



# Generate and plot one sample for each pattern type
pattern_functions = [
generate_rising_wedge,
generate_falling_wedge,
generate_symmetrical_triangle,
generate_ascending_triangle,
generate_descending_triangle,
]

# Create subplots
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
axes = axes.flatten()

for i, pattern_function in enumerate(pattern_functions):
    img, label = pattern_function()    
    # Plot the candlestick matrix
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(classes[np.argmax(label)])
    axes[i].axis('off')

plt.tight_layout()
plt.show()  