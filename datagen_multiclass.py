from numpy.random.mtrand import randint
import csv
import pathlib
import os
import pandas as pd
import numpy as np
import random as rd
from datetime import timedelta
import mplfinance as mpf


# Check if the directory exists, otherwise create it
directory = 'patterns_datasetTEST'
if not os.path.exists(directory):
    os.makedirs(directory)



classes = ["Rising Wedge", "Falling Wedge", "Symmetrical Triangle", "Ascending Triangle", "Descending Triangle" ,"No Patterns" ]



def write_to_csv(image):
    csv_file_path = str(image["path"]).split("\\")[0] + "\\patterns_dataset.csv"
    label = [1 if image["class"] == c else 0 for c in classes]
    if not os.path.isfile(csv_file_path):
        with open(csv_file_path, mode='w', newline='') as label_file:
            label_writer = csv.writer(label_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            label_writer.writerow(['filename', classes[0], classes[1], classes[2], classes[3], 
                                   classes[4], classes[5]])
    
    filename = str(image["path"]).split("\\")[1]
    with open(csv_file_path, mode='a', newline='') as label_file:
        label_writer = csv.writer(label_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        label_writer.writerow([filename,  label[0], label[1], label[2], label[3], 
                                label[4], label[5]])


def create_image(open_data, high_data, low_data, close_data,path):
    base = pd.to_datetime('2023-01-01')
    dates = [base + timedelta(days=i) for i in range(len(open_data))]

    df = pd.DataFrame({'Date': dates, 'Open': open_data, 'High': high_data, 'Low': low_data, 'Close': close_data})
    df.set_index('Date', inplace=True)
    custom_style = mpf.make_mpf_style(base_mpf_style='charles', gridstyle='')
    figsize_multiplier = 0.5
    fig = mpf.plot(df, type='candle', style=custom_style, title='', ylabel='Price', axisoff=True,  figsize=(len(open_data) * figsize_multiplier*0.3, 6*0.3), savefig=path, scale_padding=0)


def check_high_low(close_price, open_price, high_price, low_price):
    if close_price > open_price : 
              high_price = high_price if high_price > close_price else close_price
              low_price = low_price if low_price < open_price else open_price
    if close_price < open_price : 
        high_price = high_price if high_price > open_price else open_price
        low_price = low_price if low_price < close_price else close_price
        
    return high_price, low_price
  

def generate_rising_wedge(index=0):
    num_data_points = rd.randint(6, 30)
    lower_slope = np.random.uniform(0.1, 2)
    upper_slope = np.random.uniform(0.1, lower_slope - 0.2)  # Ensuring upper slope is less than lower slope
    bias = np.random.uniform(0.1, 0.5)

    x = np.linspace(0, num_data_points, num_data_points) # Generate x values
    lower_line = lower_slope * x
    upper_line = upper_slope * (x - num_data_points) + lower_line[-1] + bias

    open_data, close_data, high_data, low_data  = [], [], [], []
    split =  num_data_points//3
    start, mid, end = x[0:split], x[split :split*2], x[split*2:]
    ltouches = np.concatenate([np.where(x == rd.choice(start))[0], np.where(x == rd.choice(mid))[0], np.where(x == rd.choice(end))[0]]).tolist()
    utouches = np.concatenate([np.where(x == rd.choice(start))[0], np.where(x == rd.choice(mid))[0], np.where(x == rd.choice(end))[0]]).tolist()


    current_price = (upper_line[0] + lower_line[0]) / 2
    for i in range(num_data_points):
        open_price = current_price

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

        high_price, low_price = check_high_low(close_price, open_price, high_price, low_price)
            
        open_data.append(open_price)
        close_data.append(close_price)
        high_data.append(high_price)
        low_data.append(low_price)

        current_price = close_price  # Set the current price for the next iteration

    path = pathlib.Path(directory) / f"r_wedge{index}.jpg"
    create_image(open_data, high_data, low_data, close_data, path)
    
    return {"path":path, "class":"Rising Wedge"}  


def generate_falling_wedge(index=0):
    num_data_points = rd.randint(6, 30)
    upper_slope = -np.random.uniform(0.1, 2)
    lower_slope = np.random.uniform(-0.1, upper_slope +0.2)  # Ensuring upper slope is less than lower slope
    bias = -np.random.uniform(0.1, 0.5)
    
    x = np.linspace(0, num_data_points, num_data_points) # Generate x values
    upper_line = upper_slope * x
    lower_line = lower_slope * (x - num_data_points) + upper_line[-1] + bias
    open_data, close_data, high_data, low_data  = [], [], [], []

    split =  num_data_points//3
    start, mid, end = x[0:split], x[split :split*2], x[split*2:]
    ltouches = np.concatenate([np.where(x == rd.choice(start))[0], np.where(x == rd.choice(mid))[0], np.where(x == rd.choice(end))[0]]).tolist()
    utouches = np.concatenate([np.where(x == rd.choice(start))[0], np.where(x == rd.choice(mid))[0], np.where(x == rd.choice(end))[0]]).tolist()

    current_price = (upper_line[0] + lower_line[0]) / 2
    for i in range(num_data_points):
        open_price = current_price

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
          
        high_price, low_price = check_high_low(close_price, open_price, high_price, low_price)
            
        open_data.append(open_price)
        close_data.append(close_price)
        high_data.append(high_price)
        low_data.append(low_price)

        current_price = close_price  # Set the current price for the next iteration


    path = pathlib.Path(directory) / f"f_wedge{index}.jpg"
    create_image(open_data, high_data, low_data, close_data, path)
    
    return {"path":path, "class":"Falling Wedge"}   




def generate_symmetrical_triangle(index=0):
    num_data_points = rd.randint(6, 30)
    upper_slope = -np.random.uniform(0.1, 2)
    lower_slope = -upper_slope  # Ensuring upper slope is less than lower slope
    bias = -np.random.uniform(0.1, 0.5)
    
    x = np.linspace(0, num_data_points, num_data_points) # Generate x values

    upper_line = upper_slope * x
    lower_line = lower_slope * (x - num_data_points) + upper_line[-1] + bias
    open_data, close_data, high_data, low_data  = [], [], [], []

    split =  num_data_points//3
    start, mid, end = x[0:split], x[split :split*2], x[split*2:]
    ltouches = np.concatenate([np.where(x == rd.choice(start))[0], np.where(x == rd.choice(mid))[0], np.where(x == rd.choice(end))[0]]).tolist()
    utouches = np.concatenate([np.where(x == rd.choice(start))[0], np.where(x == rd.choice(mid))[0], np.where(x == rd.choice(end))[0]]).tolist()

    current_price = (upper_line[0] + lower_line[0]) / 2
    for i in range(num_data_points):
        open_price = current_price

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

        high_price, low_price = check_high_low(close_price, open_price, high_price, low_price)
            
        open_data.append(open_price)
        close_data.append(close_price)
        high_data.append(high_price)
        low_data.append(low_price)

        current_price = close_price  # Set the current price for the next iteration


    path = pathlib.Path(directory) / f"sym_tri{index}.jpg"
    create_image(open_data, high_data, low_data, close_data, path)
    
    return {"path":path, "class":"Symmetrical Triangle"}
  
  
def generate_ascending_triangle(index=0):
  num_data_points = rd.randint(6, 30)

  # Generate x values
  x = np.linspace(0, num_data_points, num_data_points)

  # Generate random slope for the ascending support line
  support_slope = np.random.uniform(0.1, 2)

  # Generate y values for the ascending support line
  support_line = support_slope * x

  # Straight resistance line
  resistance_line = np.ones_like(x) * np.max(support_line)

  # Candlestick data within the boundary lines
  open_data, close_data, high_data, low_data  = [], [], [], []

  split = num_data_points // 3
  start, mid, end = x[0:split], x[split :split*2], x[split*2:]
  ltouches = np.concatenate([np.where(x == rd.choice(start))[0], np.where(x == rd.choice(mid))[0], np.where(x == rd.choice(end))[0]]).tolist()
  utouches = np.concatenate([np.where(x == rd.choice(start))[0], np.where(x == rd.choice(mid))[0], np.where(x == rd.choice(end))[0]]).tolist()


  current_price = (support_line[0] + resistance_line[0]) / 2

  for i in range(num_data_points):
      open_price = current_price

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

      high_price, low_price = check_high_low(close_price, open_price, high_price, low_price)
            
      open_data.append(open_price)
      close_data.append(close_price)
      high_data.append(high_price)
      low_data.append(low_price)

      current_price = close_price  # Set the current price for the next iteration

  path = pathlib.Path(directory) / f"asc_triangle{index}.jpg"
  create_image(open_data, high_data, low_data, close_data, path)
  
  return {"path":path, "class":"Ascending Triangle"}
  

def generate_descending_triangle(index=0):
  num_data_points = rd.randint(6, 30)
  x = np.linspace(0, num_data_points, num_data_points)   # Generate x values

  resistance_slope = -np.random.uniform(0.1, 2)
  resistance_line = resistance_slope * x
  support_line = np.ones_like(x) * np.min(resistance_line)

  open_data, close_data, high_data, low_data  = [], [], [], []

  split = num_data_points // 3
  start, mid, end = x[0:split], x[split:split*2], x[split*2:]
  ltouches = np.concatenate([np.where(x == np.random.choice(start))[0], np.where(x == np.random.choice(mid))[0], np.where(x == np.random.choice(end))[0]]).tolist()
  utouches = np.concatenate([np.where(x == np.random.choice(start))[0], np.where(x == np.random.choice(mid))[0], np.where(x == np.random.choice(end))[0]]).tolist()

  current_price = (support_line[0] + resistance_line[0]) / 2

  for i in range(num_data_points):
      open_price = current_price

      if i in ltouches:
          close_or_low = np.random.choice([True, False])
          if close_or_low:
              close_price = support_line[i]
              low_price = np.random.uniform(support_line[i], resistance_line[i])
              high_price = support_line[i]
          else:
              close_price = np.random.uniform(support_line[i], resistance_line[i])
              low_price = support_line[i]
              high_price = np.random.uniform(support_line[i], resistance_line[i])
      elif i in utouches:
          close_or_high = np.random.choice([True, False])
          if close_or_high:
              close_price = resistance_line[i]
              low_price = np.random.uniform(support_line[i], resistance_line[i])
              high_price = np.random.uniform(support_line[i], resistance_line[i])
          else:
              close_price = np.random.uniform(support_line[i], resistance_line[i])
              low_price = np.random.uniform(support_line[i], resistance_line[i])
              high_price = resistance_line[i]
      else:
          close_price = np.random.uniform(support_line[i], resistance_line[i])
          low_price = np.random.uniform(support_line[i], resistance_line[i])
          high_price = np.random.uniform(support_line[i], resistance_line[i])
          
      high_price, low_price = check_high_low(close_price, open_price, high_price, low_price)
            
      open_data.append(open_price)
      close_data.append(close_price)
      high_data.append(high_price)
      low_data.append(low_price)

      current_price = close_price  # Set the current price for the next iteration

  path = pathlib.Path(directory) / f"desc_triangle{index}.jpg"
  create_image(open_data, high_data, low_data, close_data, path)
  
  return {"path":path, "class":"Descending Triangle"}   


def generate_no_pattern(index=0):
    choice = rd.choice([True, False])
    if choice: num_data_points = rd.randint(1, 6)
    else: num_data_points = rd.randint(6, 30)
        
    # Generate random prices for the candlestick data
    open_data = np.random.randint(100, 500, num_data_points)
    close_data = np.random.randint(100, 500, num_data_points)
    high_data = np.random.randint(100, 500, num_data_points)
    low_data = np.random.randint(100, 500, num_data_points)

    for i, (open_price, high_price, low_price, close_price) in enumerate(zip(open_data, high_data, low_data, close_data)):
        if close_price > open_price:
            high_data[i] = max(high_price, close_price)
            low_data[i] = min(low_price, open_price)
        elif close_price < open_price:
            high_data[i] = max(high_price, open_price)
            low_data[i] = min(low_price, close_price)
            
    path = pathlib.Path(directory) / f"no_pattern_{index}.jpg"
    create_image(open_data, high_data, low_data, close_data, path)
    return {"path": path, "class": "No Patterns"}
  


for i in range(500):
    write_to_csv(generate_rising_wedge(i))    
    write_to_csv(generate_falling_wedge(i))
    write_to_csv(generate_symmetrical_triangle(i))
    write_to_csv(generate_ascending_triangle(i))
    write_to_csv(generate_descending_triangle(i))
    write_to_csv(generate_no_pattern(i))

    
    


    