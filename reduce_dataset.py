import typer
import json
import random
from pathlib import Path
from collections import defaultdict


def process_index_file(file_path: Path, number_of_frames_to_keep: int):
    list_of_ids = defaultdict(list)

    with file_path.open("r") as file:
        data = json.load(file)

    print("Converting...\n")
    for frame in data["frames"]:
        for annotation in frame["annotations"]:
            for label in annotation["labels"]:
                list_of_ids[frame["datasetFrameId"]].append(label)

    classes_to_keep = ["person", "car", "bike"]
    filtered_dict = {}

    for key, values in list_of_ids.items():
        if all(val in classes_to_keep for val in values):
            filtered_dict[key] = values

    list_of_ids.clear()
    list_of_ids.update(filtered_dict)

    for key, value in list_of_ids.items():
        print(f"{key}: {value}")

    keys_to_save = random.sample(filtered_dict.keys(), number_of_frames_to_keep)

    for frame in data["frames"][:]:
        if frame["datasetFrameId"] not in keys_to_save:
            data["frames"].remove(frame)

    with file_path.open("w") as file:
        json.dump(data, file, indent=4)

    return keys_to_save


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
        help="Path to the json",
    ),
    dir_path: str = typer.Argument(
        None,
        help="Path to the json",
    ),
    number_of_frames_to_keep: int = typer.Argument(
        None,
        help="Path to the json",
    ),
):

    delete_files(dir_path, process_index_file(json_path, number_of_frames_to_keep))


if __name__ == "__main__":
    app()
