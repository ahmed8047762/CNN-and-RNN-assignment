import glob
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import zipfile

# Step 1: Merge and extract images from ZIP files into a simple folder
merged_images_folder = 'merged_images'

# Create the merged images folder if it doesn't exist
if not os.path.exists(merged_images_folder):
    os.makedirs(merged_images_folder)

# Extract images from HAM10000_images_part1.zip
with zipfile.ZipFile('HAM10000_images_part_1.zip', 'r') as zip_ref:
    zip_ref.extractall(merged_images_folder)
    
print("Extracted HAM10000_images_part_1.zip to", merged_images_folder)

# Extract images from HAM10000_images_part2.zip
with zipfile.ZipFile('HAM10000_images_part_2.zip', 'r') as zip_ref:
    zip_ref.extractall(merged_images_folder)
    
print("Extracted HAM10000_images_part_2.zip to", merged_images_folder)

# Step 2: Load and preprocess data
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob.glob(os.path.join(merged_images_folder, '*.jpg'))}

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

# Load metadata CSV from the root folder
skin_df = pd.read_csv('HAM10000_metadata.csv')

# Create 'path' column by mapping image_id to the corresponding file path
skin_df['path'] = skin_df['image_id'].map(imageid_path_dict.get)

# Create 'cell_type' and 'cell_type_idx' columns based on the 'dx' column
skin_df['cell_type'] = skin_df['dx'].map(lesion_type_dict.get) 
skin_df['cell_type_idx'] = pd.Categorical(skin_df['cell_type']).codes

print("Total number of lesions: ", skin_df.shape[0])
#print(skin_df.head())


# Step 2: Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(skin_df['path'],
                                                    skin_df['cell_type_idx'],
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=skin_df['cell_type_idx'])

# Step 3: Load and preprocess images

def load_images(image_paths, image_size=(100, 75)):
    images = []
    for img_path in image_paths:
        # Handle None values in image_paths
        if img_path is None:
            continue
        img = load_img(img_path, target_size=image_size)
        img_array = img_to_array(img)
        images.append(img_array)
    return np.array(images)

# Filter out None values from X_train and X_test (in case some paths are None)
X_train = [path for path in X_train if path is not None]
X_test = [path for path in X_test if path is not None]

X_train_images = load_images(X_train)
X_test_images = load_images(X_test)

# Print number of loaded images for debugging
print("Number of loaded training images:", len(X_train_images))
print("Number of loaded testing images:", len(X_test_images))

# Ensure that there are some images loaded, if not, investigate the file paths and loading logic.
if len(X_train_images) == 0 or len(X_test_images) == 0:
    print("No images loaded. Please check file paths and loading logic.")
    exit()

# Step 4: Preprocess labels

y_train_encoded = to_categorical(y_train, num_classes=7)
y_test_encoded = to_categorical(y_test, num_classes=7)

# Step 5: Build CNN model

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 75, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(7, activation='softmax')  # 7 output classes
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 6: Train the model

checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

history = model.fit(X_train_images, y_train_encoded, epochs=10, batch_size=32, validation_split=0.2, callbacks=[checkpoint])

# Step 7: Display training error

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Step 8: Evaluate the model on test data

model.load_weights('best_model.h5')  # Load the best model saved during training

y_pred = model.predict(X_test_images)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test_encoded, axis=1)

# Step 9: Display confusion matrix

confusion_mtx = confusion_matrix(y_true, y_pred_classes)
plt.imshow(confusion_mtx, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks(np.arange(7), lesion_type_dict.values(), rotation=90)
plt.yticks(np.arange(7), lesion_type_dict.values())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Step 10: Calculate accuracy, precision, recall, and f1-score

classification_rep = classification_report(y_true, y_pred_classes, target_names=lesion_type_dict.values())
print(classification_rep)

# Step 11: Save and load the model for deployment

model.save('skin_cancer_cnn_model.h5')