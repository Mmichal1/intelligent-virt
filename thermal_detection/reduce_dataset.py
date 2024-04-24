import typer
import json
import random
import numpy
import tensorflow as tf
import io

from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import cv2
import os
import glob


def are_disjoint(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    return set1.isdisjoint(set2)


def create_tf_example(image_path, annotations):
    # Load image data with OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    height, width, _ = img.shape

    # Encode the image as jpeg
    _, encoded_image_data = cv2.imencode(".jpg", img)
    encoded_jpg_io = io.BytesIO(encoded_image_data.tobytes())

    xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box (1 per box)
    ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box (1 per box)
    classes_text = []  # List of string class name of bounding box (1 per box)
    classes = []  # List of integer class id of bounding box (1 per box)

    for annotation in annotations:
        if "person" in annotation["labels"]:
            xmins.append(annotation["boundingBox"]["x"] / width)
            xmaxs.append((annotation["boundingBox"]["x"] + annotation["boundingBox"]["w"]) / width)
            ymins.append(annotation["boundingBox"]["y"] / height)
            ymaxs.append((annotation["boundingBox"]["y"] + annotation["boundingBox"]["h"]) / height)
            classes_text.append("person".encode("utf8"))
            classes.append(1)

    tf_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/height": tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                "image/width": tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                "image/filename": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[os.path.basename(image_path).encode("utf8")])
                ),
                "image/source_id": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[os.path.basename(image_path).encode("utf8")])
                ),
                "image/encoded": tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_jpg_io.getvalue()])),
                "image/format": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"jpeg"])),
                "image/object/bbox/xmin": tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
                "image/object/bbox/xmax": tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
                "image/object/bbox/ymin": tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
                "image/object/bbox/ymax": tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
                "image/object/class/text": tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
                "image/object/class/label": tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
            }
        )
    )
    return tf_example


def resize_all_frames(json_path, output_json_path, data_directory, output_directory, scale_factor=2):
    # Load the JSON data
    with json_path.open("r") as file:
        data = json.load(file)

    # Process each frame in the JSON data
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
        # Make sure the output directory exists
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        output_image_path = os.path.join(output_directory, f"resized-{os.path.basename(image_path)}")

        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Image file {image_path} could not be read.")
            continue

        # Resize the image and calculate new size
        original_size = (img.shape[1], img.shape[0])  # (width, height)
        new_size = (original_size[0] // scale_factor, original_size[1] // scale_factor)
        resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

        # Update frame width and height in JSON
        frame["width"], frame["height"] = new_size

        # Filter and scale bounding boxes directly in the JSON data structure
        new_annotations = []
        for annotation in frame["annotations"]:
            if "person" in annotation["labels"]:  # Check if 'person' is in labels
                bbox = annotation["boundingBox"]
                bbox["x"] //= scale_factor
                bbox["y"] //= scale_factor
                bbox["w"] //= scale_factor
                bbox["h"] //= scale_factor
                new_annotations.append(annotation)
        frame["annotations"] = new_annotations

        # Save the resized image
        cv2.imwrite(output_image_path, resized_img)
        print(f"Resized image saved: {output_image_path}")

    # Save the updated JSON data
    with output_json_path.open("w") as file:
        json.dump(data, file, indent=4)
    print(f"Updated JSON saved: {output_json_path}")


def process_index_file(
    file_path: Path,
    number_of_frames_with_class: int,
    number_of_frames_without_class: int,
):
    object_class_by_frame_id: Dict[str, List[str]] = defaultdict(list)

    with file_path.open("r") as file:
        data = json.load(file)

    print("Converting...\n")
    for frame in data["frames"]:
        for annotation in frame["annotations"]:
            for label in annotation["labels"]:
                object_class_by_frame_id[frame["datasetFrameId"]].append(label)

    classes_to_keep = ["person"]
    frames_containing_class: List[str] = []
    frames_not_containing_class: List[str] = []

    for frame_id, class_list in object_class_by_frame_id.items():
        if are_disjoint(class_list, classes_to_keep):
            frames_not_containing_class.append(frame_id)
        else:
            frames_containing_class.append(frame_id)

    frames_to_save = random.sample(frames_containing_class, number_of_frames_with_class) + random.sample(
        frames_not_containing_class, number_of_frames_without_class
    )

    for frame in data["frames"][:]:
        if frame["datasetFrameId"] not in frames_to_save:
            data["frames"].remove(frame)

    with file_path.open("w") as file:
        json.dump(data, file, indent=4)

    return frames_to_save


def delete_files(directory, keys_to_keep):
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if any(key in filename for key in keys_to_keep):
            print(f"Keeping file: {filename}")
        else:
            os.remove(filepath)
            print(f"Deleted file: {filename}")


app = typer.Typer(add_completion=False)


@app.command()
def main(
    json_path: Path = typer.Argument(
        None,
        help="Path to json",
    ),
    # json_output_path: Path = typer.Argument(
    #     None,
    #     help="Path to json",
    # ),
    tfrecord_output_path: Path = typer.Argument(
        None,
        help="Path to json",
    ),
    data_path: str = typer.Argument(
        None,
        help="Path to data",
    ),
    # data_output_path: str = typer.Argument(
    #     None,
    #     help="Path to data",
    # ),
    # number_of_frames_to_keep_class: int = typer.Argument(
    #     None,
    #     help="Number of frames to keep class",
    # ),
    # number_of_frames_to_keep_no_class: int = typer.Argument(
    #     None,
    #     help="Number of frames to keep no class",
    # ),
):

    # delete_files(
    #     dir_path,
    #     process_index_file(
    #         json_path,
    #         number_of_frames_to_keep_class,
    #         number_of_frames_to_keep_no_class,
    #     ),
    # )
    # find_largest_image(json_path)

    # resize_all_frames(json_path, json_output_path, data_path, data_output_path)

    writer = tf.io.TFRecordWriter(f"{tfrecord_output_path}.tfrecord")
    with json_path.open("r") as f:
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
        annotations = frame["annotations"]
        tf_example = create_tf_example(image_path, annotations)
        writer.write(tf_example.SerializeToString())
    writer.close()



if __name__ == "__main__":
    app()
