import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# Path to the dataset
DATASET_PATH = "path_to_flir_dataset"
ANNOTATIONS_PATH = os.path.join(DATASET_PATH, "coco_annotations.json")

# Load annotations
with open(ANNOTATIONS_PATH, "r") as f:
    annotations = json.load(f)


# Function to load images and annotations
def load_data():
    images = []
    bboxes = []
    labels = []

    for img_data in annotations["images"]:
        img_path = os.path.join(DATASET_PATH, img_data["file_name"])
        image = load_img(img_path, target_size=(256, 256))
        image = img_to_array(image)

        img_id = img_data["id"]
        img_annotations = [ann for ann in annotations["annotations"] if ann["image_id"] == img_id]

        for ann in img_annotations:
            bbox = ann["bbox"]
            label = ann["category_id"]

            images.append(image)
            bboxes.append(bbox)
            labels.append(label)

    return np.array(images), np.array(bboxes), np.array(labels)


images, bboxes, labels = load_data()

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, (bboxes, labels), test_size=0.2, random_state=42)


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense


def create_model(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), activation="relu")(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)

    # Output layers: one for bounding boxes, one for labels
    bbox_output = Dense(4, name="bbox_output")(x)
    label_output = Dense(len(annotations["categories"]), activation="softmax", name="label_output")(x)

    model = Model(inputs=inputs, outputs=[bbox_output, label_output])
    return model


input_shape = (256, 256, 3)
model = create_model(input_shape)
model.summary()

input_shape = (256, 256, 3)
model = create_model(input_shape)
model.summary()

model.compile(
    optimizer="adam",
    loss={"bbox_output": "mse", "label_output": "sparse_categorical_crossentropy"},
    metrics={"bbox_output": "mae", "label_output": "accuracy"},
)

history = model.fit(
    X_train,
    {"bbox_output": y_train[0], "label_output": y_train[1]},
    validation_data=(X_val, {"bbox_output": y_val[0], "label_output": y_val[1]}),
    epochs=20,
    batch_size=32,
)

val_loss, val_bbox_loss, val_label_loss, val_bbox_mae, val_label_accuracy = model.evaluate(
    X_val, {"bbox_output": y_val[0], "label_output": y_val[1]}
)

print(f"Validation bbox MAE: {val_bbox_mae}")
print(f"Validation label accuracy: {val_label_accuracy}")
