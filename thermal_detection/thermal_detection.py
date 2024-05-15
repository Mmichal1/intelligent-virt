import os
import json
import typer
import numpy as np
import tensorflow as tf
from pathlib import Path
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split


# Load annotations
def load_coco_data(annotations_path: Path):
    with annotations_path.open("r") as f:
        return json.load(f)


# Function to load a limited number of images and annotations for people only
def load_data(annotations_path: Path, data_path: Path, limit=None, target_size=(160, 128)):
    images = []
    bboxes = []
    labels = []
    count = 0

    coco_data = load_coco_data(annotations_path)

    for img_data in coco_data["images"]:
        if limit and count >= limit:
            break

        img_path: Path = data_path / img_data["file_name"]
        image = load_img(img_path, target_size=target_size)  # Rescale to target size
        image = img_to_array(image)

        img_id = img_data["id"]
        img_annotations = [
            ann for ann in coco_data["annotations"] if ann["image_id"] == img_id and ann["category_id"] == 3
        ]  # Assuming category_id 1 is for people

        if img_annotations:
            for ann in img_annotations:
                original_bbox = ann["bbox"]

                # Rescale bounding box
                scale_x = target_size[0] / 640
                scale_y = target_size[1] / 512
                bbox = [
                    original_bbox[0] * scale_x,
                    original_bbox[1] * scale_y,
                    original_bbox[2] * scale_x,
                    original_bbox[3] * scale_y,
                ]

                label = 1  # Person label

                images.append(image)
                bboxes.append(bbox)
                labels.append(label)
            count += 1

    return np.array(images), np.array(bboxes), np.array(labels)


def create_model(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, (3, 3), activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)

    # Output layers: one for bounding boxes, one for labels
    bbox_output = Dense(4, name="bbox_output")(x)
    label_output = Dense(1, activation="sigmoid", name="label_output")(x)

    model = Model(inputs=inputs, outputs=[bbox_output, label_output])
    return model


app = typer.Typer(add_completion=False)


@app.command()
def main(
    dataset_path: Path = typer.Argument(
        None,
        help="Path to json",
    ),
):
    # Load a limited number of images (e.g., 100 images)

    annotations_path = dataset_path / "coco.json"

    images, bboxes, labels = load_data(annotations_path, dataset_path, limit=1000)

    # Split the data into training and validation sets
    X_train, X_val, y_train_bboxes, y_val_bboxes, y_train_labels, y_val_labels = train_test_split(
        images, bboxes, labels, test_size=0.2, random_state=42
    )
    target_width: int = 160
    target_height: int = 128
    input_shape = (target_width, target_height, 3)
    model = create_model(input_shape)
    model.summary()

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss={"bbox_output": "mse", "label_output": "binary_crossentropy"},
        metrics={"bbox_output": "mae", "label_output": "accuracy"},
    )

    history = model.fit(
        X_train,
        {"bbox_output": y_train_bboxes, "label_output": y_train_labels},
        validation_data=(X_val, {"bbox_output": y_val_bboxes, "label_output": y_val_labels}),
        epochs=5,
        batch_size=100,
    )

    # Evaluate the model and print the output to see what it returns
    evaluation = model.evaluate(X_val, {"bbox_output": y_val_bboxes, "label_output": y_val_labels})
    print(f"Evaluation results: {evaluation}")


if __name__ == "__main__":
    app()
