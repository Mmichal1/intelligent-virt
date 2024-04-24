import cv2
import json
import os
import typer
import glob
import numpy as np
from pathlib import Path
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split


# Function to extract patches from the dataset
def extract_patches(json_path, data_path, patch_size=(64, 64)):
    patches = []
    labels = []

    with open(json_path, "r") as f:
        data = json.load(f)

    for frame in data["frames"]:
        dataset_frame_id = frame["datasetFrameId"]
        search_pattern = os.path.join(data_path, f"*-{dataset_frame_id}.jpg")
        image_files = glob.glob(search_pattern)

        if not image_files:
            print(f"Image file matching the dataset frame ID {dataset_frame_id} not found.")
            continue  # Skip this frame if the image file is not found
        elif len(image_files) > 1:
            print(f"Warning: Multiple files found for {dataset_frame_id}, using the first one.")

        image_path = image_files[0]
        img = cv2.imread(image_path)
        if img is None:
            continue

        for annotation in frame["annotations"]:
            if "person" in annotation["labels"]:
                x = int(annotation["boundingBox"]["x"])
                y = int(annotation["boundingBox"]["y"])
                w = int(annotation["boundingBox"]["w"])
                h = int(annotation["boundingBox"]["h"])

                # Extract the patch and resize
                patch = img[y : y + h, x : x + w]
                patch = cv2.resize(patch, patch_size)

                patches.append(patch)
                labels.append(1)  # This patch contains a person

    return patches, labels


# Function to build a simple CNN model
def build_model(input_shape):
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            Dropout(0.2),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            Dropout(0.2),
            Conv2D(64, (3, 3), activation="relu"),
            Dropout(0.2),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


app = typer.Typer(add_completion=False)


@app.command()
def main(
    json_path: Path = typer.Argument(
        None,
        help="Path to json",
    ),
    data_path: str = typer.Argument(
        None,
        help="Path to data",
    ),
):

    patches, labels = extract_patches(json_path, data_path)

    X = np.array([img_to_array(patch) for patch in patches])
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model(X_train[0].shape)
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))


if __name__ == "__main__":
    app()
