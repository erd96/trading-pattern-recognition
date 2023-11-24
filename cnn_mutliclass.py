import pandas as pd
import numpy as np
import cv2
import random
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import LearningRateScheduler
from sklearn.utils import shuffle
# Step 1: Load the data
data = pd.read_csv('patterns_dataset2/patterns_dataset.csv')

# Step 2: Preprocess the data
images = []
labels = []

for index, row in data.iterrows():
    img = cv2.imread('patterns_dataset2/' + row['filename'])
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    images.append(img.astype('float32') / 255)  # Convert to float32 and normalize
    labels.append(row[["Rising Wedge", "Falling Wedge", "Symmetrical Triangle", "Ascending Triangle", "Descending Triangle" ,"No Patterns" ]].values.astype('float32'))

images = np.array(images)
labels = np.array(labels)

plt.imshow(images[0])
plt.axis('off')
plt.show()



def augment_data(images, labels):
# Step 2.1: Image Augmentation
    datagen = ImageDataGenerator(zoom_range=[2, 3])    
    augmented_images = []
    augmented_labels = []

    for i in range(images.shape[0]):
        img = images[i]
        label = labels[i]

        augmented_images.append(img)
        augmented_labels.append(label)

        # Generate 4 zoom variants
        for j in range(2):
            augmented_img = datagen.random_transform(img)
            augmented_images.append(augmented_img)
            augmented_labels.append(label)

            # # Visualize the original and its variants
            # plt.subplot(1, 5, 1)
            # plt.imshow(img.reshape(128, 128), cmap='gray')
            # plt.title(f'Original\nLabel: {np.argmax(label)}')
            # plt.axis('off')

            # plt.subplot(1, 5, j + 2)
            # plt.imshow(augmented_img.reshape(128, 128), cmap='gray')
            # plt.title(f'Variant {j + 1}\nLabel: {np.argmax(label)}')
            # plt.axis('off')

        # plt.show()
    return np.array(augmented_images), np.array(augmented_labels)

# augmented_images = np.array(augmented_images)
# augmented_labels = np.array(augmented_labels)


# print(len(augmented_images), len(augmented_labels))
# Visualize an example of an augmented image
# plt.imshow(augmented_images[1].reshape(128, 128), cmap='gray')
# plt.axis('off')
# plt.show()



from keras.layers import Dropout

model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D((2, 2)))


model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))


model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))  # Add dropout before the final dense layer

model.add(Dense(6, activation='softmax'))

# # Step 4: Compile the model
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',  # 'categorical_crossentropy' for multiclass classification
#               metrics=['accuracy'])

# model.summary()

def lr_scheduler(epoch):
    if epoch < 10:
        return 0.001 * np.exp(-epoch / 10)
    else:
        return 0.0001 * np.exp(-(epoch-10) / 10)

lr_schedule = LearningRateScheduler(lr_scheduler)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])



# Shuffle the data
images_shuffled, labels_shuffled = shuffle(images, labels, random_state=42)

# Split the data into train, validation, and test sets
X_train, X_val_test, y_train, y_val_test = train_test_split(images_shuffled, labels_shuffled, test_size=0.3, random_state=42)
X_train, y_train = augment_data(X_train, y_train)
X_val_test, y_val_test = augment_data(X_val_test, y_val_test)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)
# Step 6: Fit the model
# model.fit(X_train, y_train, epochs=15, validation_data=(X_val, y_val))
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), callbacks=[lr_schedule])



# Convert the prediction and ground truth label to readable format
# pattern_classes = ['Double Bottom', 'Double Top', 'Head and shoulders', 'Reverse Head and shoulders',
#                    'Rising Wedge', 'Falling Wedge', 'Symmetrical Triangle', 'No Patterns']

pattern_classes = ["Rising Wedge", "Falling Wedge", "Symmetrical Triangle", "Ascending Triangle", "Descending Triangle" ,"No Patterns" ]

# Step 7: Make 10 random predictions
num_samples = 10
random_indices = random.sample(range(len(X_test)), num_samples)

plt.figure(figsize=(15, 10))
for i, sample_index in enumerate(random_indices):
    sample_image = X_test[sample_index]
    sample_label = y_test[sample_index]

    # Reshape the image for prediction
    sample_image = sample_image.reshape(1, 128, 128, 3)

    # Make a prediction
    prediction = model.predict(sample_image)
    
    print(prediction)

    # Convert the prediction and ground truth label to readable format
    prediction_label = pattern_classes[np.argmax(prediction)]
    true_label = pattern_classes[np.argmax(sample_label)]

    # Display the sample image and its corresponding prediction
    plt.subplot(2, 5, i + 1)
    plt.imshow(sample_image[0])
    plt.title(f'Prediction: {prediction_label}\nTrue Label: {true_label}')
    plt.axis('off')

plt.show()

# Step 8: Save the model
model.save('patterns_classification_model_5.h5')

