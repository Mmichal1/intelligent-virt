import typer
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List


def are_disjoint(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    return set1.isdisjoint(set2)


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


import os


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
    dir_path: str = typer.Argument(
        None,
        help="Path to data",
    ),
    number_of_frames_to_keep_class: int = typer.Argument(
        None,
        help="Number of frames to keep class",
    ),
    number_of_frames_to_keep_no_class: int = typer.Argument(
        None,
        help="Number of frames to keep no class",
    ),
):

    delete_files(
        dir_path,
        process_index_file(
            json_path,
            number_of_frames_to_keep_class,
            number_of_frames_to_keep_no_class,
        ),
    )


if __name__ == "__main__":
    app()
