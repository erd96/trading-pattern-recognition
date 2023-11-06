# import pandas as pd
# import numpy as np
# import cv2
# from keras.models import Sequential
# from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

# import matplotlib.pyplot as plt

# # Step 1: Load the data
# data = pd.read_csv('patterns_dataset/patterns_dataset.csv')

# # Step 2: Preprocess the data
# images = []
# labels = []

# for index, row in data.iterrows():
#     img = cv2.imread('patterns_dataset/' + row['filename'])
#     img = cv2.resize(img, (128, 128))
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     plt.imshow(img_gray, cmap='gray')
#     plt.show()
#     _, img_bw = cv2.threshold(img_gray, 220, 255, cv2.THRESH_BINARY)
#     images.append(img_bw)
#     labels.append(row[['Double Bottom', 'Double Top', 'Head and shoulders', 'Reverse Head and shoulders',
#                       'Rising Wedge', 'Falling Wedge', 'Symmetrical Triangle', 'No Patterns']].values)

# images = np.array(images)
# images = images.reshape(images.shape[0], 128, 128, 1)  # Add the channel dimension
# labels = np.array(labels)

# plt.imshow(images[0].reshape(128,128), cmap='gray')
# plt.axis('off')
# plt.show()

import pandas as pd
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


# Step 1: Load the data
data = pd.read_csv('patterns_dataset/patterns_dataset.csv')

# Step 2: Preprocess the data
images = []
labels = []

for index, row in data.iterrows():
    img = cv2.imread('patterns_dataset/' + row['filename'])
    img = cv2.resize(img, (128, 128))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    
    # Convert non-white pixels to black for specified filenames
    # if row['filename'].startswith(("Reverse", "Head", "Double")):
    #     img_gray[img_gray < 252] = 0
    
    images.append(img_gray.astype('float32') / 255)  # Convert to float32 and normalize
    labels.append(row[['Double Bottom', 'Double Top', 'Head and shoulders', 'Reverse Head and shoulders',
                      'Rising Wedge', 'Falling Wedge', 'Symmetrical Triangle', 'No Patterns']].values.astype('float32'))


images = np.array(images)
images = images.reshape(images.shape[0], 128, 128, 1)  # Add the channel dimension
labels = np.array(labels)

plt.imshow(images[1].reshape(128, 128), cmap='gray')
plt.axis('off')
plt.show()


# Step 3: Build the Keras model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(8, activation='softmax'))  # 'softmax' for multiclass classification

# Step 4: Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # 'categorical_crossentropy' for multiclass classification
              metrics=['accuracy'])

model.summary()
# Step 5: Split the data into train, validation, and test sets
X_train, X_val_test, y_train, y_val_test = train_test_split(images, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

# Convert labels to categorical one-hot encoding
# y_train_categorical = to_categorical(y_train, num_classes=8)
# y_val_categorical = to_categorical(y_val, num_classes=8)
# y_test_categorical = to_categorical(y_test, num_classes=8)

# Step 6: Fit the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Step 7: Make predictions
sample_index = 10  # Choose a specific index for visualization
sample_image = images[sample_index]
sample_label = labels[sample_index]

# Reshape the image for prediction
sample_image = sample_image.reshape(1, 128, 128, 1)

# Make a prediction
prediction = model.predict(sample_image)

# Convert the prediction and ground truth label to readable format
pattern_classes = ['Double Bottom', 'Double Top', 'Head and shoulders', 'Reverse Head and shoulders',
                   'Rising Wedge', 'Falling Wedge', 'Symmetrical Triangle', 'No Patterns']
prediction_label = pattern_classes[np.argmax(prediction)]
true_label = pattern_classes[np.argmax(sample_label)]

# Display the sample image and its corresponding prediction
plt.imshow(sample_image.reshape(128, 128), cmap='gray')
plt.title(f'Prediction: {prediction_label}, True Label: {true_label}')
plt.axis('off')
plt.show()


# Step 8: Save the model
model.save('patterns_classification_model.h5')


# # ... (continue with the rest of the code for model creation and training)

# # Step 3: Create the neural network architecture
# model = Sequential()
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dense(8, activation='sigmoid'))  # 'sigmoid' for multi-label classification

# # Step 4: Compile the model
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',  # 'binary_crossentropy' for multi-label classification
#               metrics=['accuracy'])

# # Step 5: Train the model
# model.fit(images, epochs=epochs, steps_per_epoch=steps_per_epoch)

# # Step 6: Evaluate the model
# loss, accuracy = model.evaluate(image_data, steps=validation_steps)
# print(f'Loss: {loss}, Accuracy: {accuracy}')
