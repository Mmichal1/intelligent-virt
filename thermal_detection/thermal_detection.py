import cv2
import json
import os
import typer
import matplotlib.pyplot as plt
import glob
import numpy as np
from pathlib import Path
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split


def load_and_label_images(json_path, data_directory):
    images = []
    labels = []

    with open(json_path, "r") as f:
        data = json.load(f)

    for frame in data["frames"]:
        dataset_frame_id = frame["datasetFrameId"]
        search_pattern = os.path.join(data_directory, f"*-{dataset_frame_id}.jpg")
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

        images.append(img)

        # Label the image as '1' if it contains at least one person
        labels.append(int(any("person" in ann["labels"] for ann in frame["annotations"])))

    return np.array(images), np.array(labels)


def build_model():
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(256, 320, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


app = typer.Typer(add_completion=False)


@app.command()
def main(
    train_json_path: Path = typer.Argument(
        None,
        help="Path to json",
    ),
    train_data_path: str = typer.Argument(
        None,
        help="Path to data",
    ),
    val_json_path: Path = typer.Argument(
        None,
        help="Path to json",
    ),
    val_data_path: str = typer.Argument(
        None,
        help="Path to data",
    ),
):

    X_train, y_train = load_and_label_images(train_json_path, train_data_path)
    X_val, y_val = load_and_label_images(val_json_path, val_data_path)

    model = build_model()
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

    # Plotting the training and validation loss
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_validation_metrics.png")  # Save the plot as a PNG file
    plt.close()  # Close the plot to free up memory


if __name__ == "__main__":
    app()
