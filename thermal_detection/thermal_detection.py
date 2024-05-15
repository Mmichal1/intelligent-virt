import json
import typer
import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.patches import Rectangle
from pathlib import Path
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.utils import Sequence
from keras._tf_keras.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout


def load_coco_data(annotations_path: Path):
    with annotations_path.open("r") as f:
        return json.load(f)


def resize_bbox(original_bbox, scale_down_factor):
    return [
        int(original_bbox[0] / scale_down_factor) - 2,
        int(original_bbox[1] / scale_down_factor) - 2,
        int(original_bbox[2] / scale_down_factor) + 2,
        int(original_bbox[3] / scale_down_factor) + 2,
    ]


class DataGenerator(Sequence):
    def __init__(
        self,
        annotations_path,
        data_path,
        scale_down_factor,
        batch_size=32,
        target_size=(160, 128),
        limit=None,
        shuffle=True,
        **kwargs,
    ):
        self.annotations_path = annotations_path
        self.data_path = data_path
        self.scale_down_factor = scale_down_factor
        self.batch_size = batch_size
        self.target_size = target_size
        self.coco_data = load_coco_data(annotations_path)
        self.image_ids = [img["id"] for img in self.coco_data["images"][:limit]]
        self.shuffle = shuffle
        self.on_epoch_end()
        super().__init__(**kwargs)

    def __len__(self):
        return math.ceil(len(self.image_ids) / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_ids)

    def __getitem__(self, index):
        batch_image_ids = self.image_ids[index * self.batch_size : (index + 1) * self.batch_size]
        images, bboxes, labels = self.__data_generation(batch_image_ids)
        return np.array(images), {"bbox_output": np.array(bboxes), "label_output": np.array(labels)}

    def __data_generation(self, batch_image_ids):
        images = []
        bboxes = []
        labels = []

        for img_id in batch_image_ids:
            img_data = next(img for img in self.coco_data["images"] if img["id"] == img_id)
            img_path = self.data_path / img_data["file_name"]
            image = load_img(img_path, target_size=self.target_size)
            image = img_to_array(image)
            image /= 255.0

            img_annotations = [
                ann
                for ann in self.coco_data["annotations"]
                if ann["image_id"] == img_id
                and ann["category_id"] == 3
                and "occluded" in ann["extra_info"]
                and ann["extra_info"]["occluded"] == "no_(fully_visible)"
                and "truncated" not in ann["extra_info"]
            ]  # Assuming category_id 3 is for cars

            if img_annotations:
                for ann in img_annotations:
                    original_bbox = ann["bbox"]
                    bbox = resize_bbox(original_bbox, self.scale_down_factor)

                    label = 1  # Person label

                    images.append(image)
                    bboxes.append(bbox)
                    labels.append(label)

        return images, bboxes, labels


def display_image_with_bboxes(image, bbox):
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], edgecolor="r", facecolor="none")
    ax.add_patch(rect)

    plt.show()


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

    x = Conv2D(256, (3, 3), activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)

    bbox_output = Dense(4, name="bbox_output")(x)
    label_output = Dense(1, activation="sigmoid", name="label_output")(x)

    model = Model(inputs=inputs, outputs=[bbox_output, label_output])
    return model


app = typer.Typer(add_completion=False)


@app.command()
def main(
    data_path: Path = typer.Argument(
        None,
        help="Path to dataset directory containing images and coco.json",
    ),
    limit: int = typer.Option(100, help="Limit the number of images to load"),
    epochs: int = typer.Option(20, help="Number of epochs to train"),
    batch_size: int = typer.Option(32, help="Batch size for training"),
    show_image: bool = typer.Option(False, help="Show one random image with bounding boxes"),
    scale_down_factor: int = typer.Option(2, help="How much to scale down images"),
):
    annotations_path = data_path / "coco.json"
    dataset_path = data_path

    target_width = int(640 / scale_down_factor)
    target_height = int(512 / scale_down_factor)
    target_size = (target_height, target_width)
    input_shape = (target_height, target_width, 3)

    if show_image:
        data_gen = DataGenerator(
            annotations_path,
            dataset_path,
            scale_down_factor,
            batch_size=1,
            target_size=target_size,
            limit=limit,
            shuffle=True,
        )
        images, bboxes_labels = data_gen.__getitem__(0)
        images = images[0]
        bboxes = bboxes_labels["bbox_output"][0]
        display_image_with_bboxes(images, bboxes)

    model = create_model(input_shape)
    model.summary()

    model.compile(
        optimizer=Adam(learning_rate=1e-4),  # type: ignore
        loss={"bbox_output": "mse", "label_output": "binary_crossentropy"},
        metrics={"label_output": "accuracy"},
    )

    train_gen = DataGenerator(
        annotations_path,
        data_path,
        scale_down_factor,
        batch_size=batch_size,
        target_size=target_size,
        limit=limit,
        shuffle=True,
    )
    val_gen = DataGenerator(
        annotations_path,
        data_path,
        scale_down_factor,
        batch_size=batch_size,
        target_size=target_size,
        limit=limit,
        shuffle=False,
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
    )

    evaluation = model.evaluate(val_gen)
    print(f"Evaluation results: {evaluation}")

    print("Training and Validation Metrics:")
    for key in history.history.keys():
        print(f"{key}: {history.history[key]}")

    plot_metrics(history)


def plot_metrics(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["label_output_accuracy"])
    plt.plot(history.history["val_label_output_accuracy"])
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")

    plt.savefig("training_validation_metrics.png")
    # plt.show()


if __name__ == "__main__":
    app()
