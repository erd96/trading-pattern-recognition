from numpy.random.mtrand import randint
import csv
import pathlib
import os
import pandas as pd
import numpy as np
import random as rd
from datetime import timedelta
import mplfinance as mpf
from PIL import Image




def generate_rising_wedge(index=0):
    num_data_points = rd.randint(6, 20)
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


        open_data.append(open_price)
        close_data.append(close_price)
        high_data.append(high_price)
        low_data.append(low_price)

        current_price = close_price  # Set the current price for the next iteration

    base = pd.to_datetime('2023-01-01')
    dates = [base + timedelta(days=i) for i in range(num_data_points)]

    df = pd.DataFrame({'Date': dates, 'Open': open_data, 'High': high_data, 'Low': low_data, 'Close': close_data})
    df.set_index('Date', inplace=True)
    custom_style = mpf.make_mpf_style(base_mpf_style='charles', gridstyle='')
    figsize_multiplier = 0.5
    path = pathlib.Path("wedges_triangles_data") / f"r_wedge{index}.png"
    fig = mpf.plot(df, type='candle', style=custom_style, title='', ylabel='Price', axisoff=True,  figsize=(num_data_points * figsize_multiplier*0.3, 6*0.3), savefig=path, scale_padding=0)
    return {"path":path, "class":"Rising Wedge"}  


def generate_falling_wedge(index=0):
    num_data_points = rd.randint(6, 20)
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


        open_data.append(open_price)
        close_data.append(close_price)
        high_data.append(high_price)
        low_data.append(low_price)

        current_price = close_price  # Set the current price for the next iteration

# Generate random dates
    base = pd.to_datetime('2023-01-01')
    dates = [base + timedelta(days=i) for i in range(num_data_points)]

    df = pd.DataFrame({'Date': dates, 'Open': open_data, 'High': high_data, 'Low': low_data, 'Close': close_data})

    df.set_index('Date', inplace=True)
    custom_style = mpf.make_mpf_style(base_mpf_style='charles', gridstyle='')
    figsize_multiplier = 0.5
    path = pathlib.Path("wedges_triangles_data") / f"f_wedge{index}.png"
    fig = mpf.plot(df, type='candle', style=custom_style, title='', ylabel='Price', axisoff=True,  figsize=(num_data_points * figsize_multiplier*0.3, 6*0.3), savefig=path, scale_padding=0)
    return {"path":path, "class":"Falling Wedge"}   




def generate_symmetrical_triangle(index=0):
    num_data_points = rd.randint(6, 20)
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


        open_data.append(open_price)
        close_data.append(close_price)
        high_data.append(high_price)
        low_data.append(low_price)

        current_price = close_price  # Set the current price for the next iteration

# Generate random dates
    base = pd.to_datetime('2023-01-01')
    dates = [base + timedelta(days=i) for i in range(num_data_points)]

    df = pd.DataFrame({'Date': dates, 'Open': open_data, 'High': high_data, 'Low': low_data, 'Close': close_data})

    df.set_index('Date', inplace=True)
    custom_style = mpf.make_mpf_style(base_mpf_style='charles', gridstyle='')
    figsize_multiplier = 0.5
    path = pathlib.Path("wedges_triangles_data") / f"sym_tri{index}.png"
    fig = mpf.plot(df, type='candle', style=custom_style, title='', ylabel='Price', axisoff=True,  figsize=(num_data_points * figsize_multiplier*0.3, 6*0.3), savefig=path, scale_padding=0)
    return {"path":path, "class":"Symmetrical Triangle"}
    


def write_to_csv(image):
    csv_file_path = str(image["path"]).split("\\")[0] + "\\patterns_dataset.csv"
    
    if not os.path.isfile(csv_file_path):
        with open(csv_file_path, mode='w', newline='') as label_file:
            label_writer = csv.writer(label_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            label_writer.writerow(['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])

    
    with Image.open(image["path"]) as img:
        width, height = img.size
        filename = str(image["path"]).split("\\")[1]
        class_name = image["class"]
        xmin, ymin = 0, 0
        xmax, ymax = width, height
        with open(csv_file_path, mode='a', newline='') as label_file:
            label_writer = csv.writer(label_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            label_writer.writerow([filename, width, height, class_name, xmin, ymin, xmax, ymax])



    


# Check if the directory exists, otherwise create it
directory = 'wedges_triangles_data'
if not os.path.exists(directory):
    os.makedirs(directory)


for i in range(300):
    write_to_csv(generate_rising_wedge(i))    
    write_to_csv(generate_falling_wedge(i))
    write_to_csv(generate_symmetrical_triangle(i))
    


    