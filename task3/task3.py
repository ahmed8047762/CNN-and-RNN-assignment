import numpy as np
import seaborn as sns
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler

pic_size = 48
base_path = "C:/Users/admin/Desktop/Assignment2/task3/images/images/"

# Display some images for every different expression
plt.figure(0, figsize=(12,20))
cpt = 0

for expression in os.listdir(base_path + "train"):
    for i in range(1,6):
        cpt = cpt + 1
        plt.subplot(7,5,cpt)
        img = load_img(base_path + "train/" + expression + "/" +os.listdir(base_path + "train/" + expression)[i], target_size=(pic_size, pic_size))
        plt.imshow(img, cmap="gray")

plt.tight_layout()
plt.show()

# Image augmentation using keras ImageDataGenerator
batch_size = 128
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=20,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    base_path + "train",
    target_size=(56, 56),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    base_path + "validation",
    target_size=(56, 56),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Task 1: Exploratory Data Analysis
# Visualize the distribution of facial expression labels
train_class_distribution = train_generator.classes
train_class_labels = list(train_generator.class_indices.keys())

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.countplot(y=train_class_distribution, order=np.arange(len(train_class_labels)), palette="viridis")
plt.title('Class Distribution')
plt.xlabel('Count')
plt.ylabel('Expression Labels')
plt.yticks(np.arange(len(train_class_labels)), train_class_labels)
plt.show()

# Display 4 random images from each expression class
plt.figure(0, figsize=(12, 20))
cpt = 0
for expression in os.listdir(base_path + "train"):
    for i in range(1, 5):
        cpt = cpt + 1
        plt.subplot(7, 4, cpt)
        img_path = np.random.choice(os.listdir(base_path + "train/" + expression))
        img = load_img(base_path + "train/" + expression + "/" + img_path, target_size=(pic_size, pic_size))
        plt.imshow(img, cmap="gray")
        plt.xlabel(expression)
plt.tight_layout()
plt.show()

# Task 2: Model Building and Optimization
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(56, 56, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Task 3: Model Training and Evaluation
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_generator.n // batch_size
)

# Task 4: Model Evaluation using appropriate metrics
target_names = train_class_labels
Y_pred = model.predict(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

# Task 5: Export and Save the Model
model.save('task3/expression_recognition_model.h5')
print('Model saved successfully.')