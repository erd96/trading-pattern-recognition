import random as rd
import requests
import pandas as pd
import mplfinance as mpf
import io
import cv2
import numpy as np


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

    # Generate x values

    print(response)

    # Candlestick data within the boundary lines
    dates = [pd.to_datetime(entry[0], unit='ms') for entry in response]
    open_data = [float(entry[1]) for entry in response]
    close_data = [float(entry[2]) for entry in response]
    high_data = [float(entry[3]) for entry in response]
    low_data = [float(entry[4]) for entry in response]

    df = pd.DataFrame({'Date': dates, 'Open': open_data, 'High': high_data, 'Low': low_data, 'Close': close_data})

    df.set_index('Date', inplace=True)
    custom_style = mpf.make_mpf_style(base_mpf_style='charles', gridstyle='', marketcolors=mpf.make_marketcolors(
        up='black', down='black', edge='black', wick='black'))
    figsize_multiplier = 0.5
    # path = pathlib.Path(directory) / f"sym_tri{index}.jpg"
    fig, axlist = mpf.plot(df, type='candle', style=custom_style, title='', ylabel='Price', axisoff=True,   scale_padding=0, returnfig=True)
    
    # Save the figure to BytesIO
    img_bytes_io = io.BytesIO()
    fig.savefig(img_bytes_io, format='png')
    img_bytes_io.seek(0)

    return img_bytes_io


def preprocess_image(img):
    img.seek(0)
    img_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
    img_cv2 = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    # Convert the image to single-channel grayscale
    img_gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    
    # Resize the image to the target size and specify the number of channels as 1
    img_gray_resized = cv2.resize(img_gray, (128, 128), interpolation=cv2.INTER_AREA)
    img_gray_resized = img_gray_resized.reshape((*img_gray_resized.shape, 1))
    img_gray_resized = img_gray_resized.astype('float32') / 255  # Convert to float32 and normalize
    return img_gray_resized



chart_image = preprocess_image(generate_bitcoin_chart(20))

# cv2.imshow('Bitcoin Chart - Grayscale',chart_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
