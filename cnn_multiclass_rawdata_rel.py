import numpy as np
import cv2
import random
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import LearningRateScheduler
from sklearn.utils import shuffle
import datagen_mc_rawdata_rel as datagen


# Initialize empty arrays to store data and labels
img_array = []
label_array = []

# Specify the number of samples for each class
num_samples_per_class = 10000

# Generate and store samples for each class
for i in range(num_samples_per_class):
    img, label = datagen.generate_rising_wedge()
    img = datagen.pad_image(img)
    img_array.append(img/255)
    label_array.append(label)

    img, label = datagen.generate_falling_wedge()
    img = datagen.pad_image(img)
    img_array.append(img/255)
    label_array.append(label)

    img, label = datagen.generate_symmetrical_triangle()
    img = datagen.pad_image(img)
    img_array.append(img/255)
    label_array.append(label)

    img, label = datagen.generate_ascending_triangle()
    img = datagen.pad_image(img)
    img_array.append(img/255)
    label_array.append(label)

    img, label = datagen.generate_descending_triangle()
    img = datagen.pad_image(img)
    img_array.append(img/255)
    label_array.append(label)


img_array = datagen.resize_images(img_array, (32,32))
label_array = np.array(label_array)

pattern_classes = ["Rising Wedge", "Falling Wedge", "Symmetrical Triangle", "Ascending Triangle", "Descending Triangle"]
print(pattern_classes[np.argmax(label_array[1])])
cv2.imshow(f'{pattern_classes[np.argmax(label_array[1])]}', img_array[1])
cv2.waitKey(0)
cv2.destroyAllWindows()



model = Sequential()

# Convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(2, 2))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(2, 2))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(2, 2))


# Flatten layer to transition from convolutional to dense layers
model.add(Flatten())

# Dense (fully connected) layers with dropout
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))  # Adding dropout for regularization
model.add(Dense(64, activation='relu'))


# Output layer with softmax activation (since it's a multi-class classification problem)
model.add(Dense(5, activation='softmax'))




def lr_scheduler(epoch):
    if epoch < 10:
        return 0.001 * np.exp(-epoch / 10)
    else:
        return 0.0001 * np.exp(-(epoch-10) / 10)

lr_schedule = LearningRateScheduler(lr_scheduler)

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])



# images_shuffled, labels_shuffled = shuffle(img_array, label_array, random_state=42)

# Split the data into train, validation, and test sets
X_train, X_val_test, y_train, y_val_test = train_test_split(img_array, label_array, test_size=0.3, random_state=42)

# Further split the validation/test set into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)


# Step 6: Fit the model
# model.fit(X_train, y_train, epochs=15, validation_data=(X_val, y_val))
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))


pattern_classes = ["Rising Wedge", "Falling Wedge", "Symmetrical Triangle", "Ascending Triangle", "Descending Triangle"]




prediction = model.predict(X_test[0].reshape(1, 32, 32, 1))

print("Raw prediction: ", prediction, "AM Pred: ", pattern_classes[np.argmax(prediction)], "Label: ", y_test[0])


# # Step 8: Save the model
model.save('patterns_classification_model_rawdata_rel.h5')

