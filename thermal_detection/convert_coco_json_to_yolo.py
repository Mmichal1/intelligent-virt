from glob import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import matplotlib.patches as patches
import cv2
from tqdm import tqdm


# images_thermal = glob("/home/michal/Desktop/IntelligentVirtualization/FLIR_ADAS_v2/images_thermal_train/images/*.jpg")


# print(len(images_thermal))

# # plot multiple random thermal images
# ran_gen = np.random.default_rng()

# plt.figure(figsize=(16, 14))
# plt.suptitle("Thermal Images")
# for i in range(12):
#     ax = plt.subplot(4, 4, i + 1)
#     random_index = ran_gen.integers(low=0, high=3748, size=1)
#     i = random_index[0]
#     img_loc = images_thermal[i]
#     img_title = (
#         "video: "
#         + images_thermal[i][-52:-35]
#         + "\n"
#         + "frame: "
#         + images_thermal[i][-28:-22]
#         + "\n"
#         + "id: "
#         + images_thermal[i][-21:-4]
#     )
#     image = plt.imread(img_loc)
#     plt.imshow(image, cmap="binary")
#     plt.title(img_title, fontsize="small")
#     plt.axis(False)

# plt.show()


def make_folders(output_path):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    return output_path


def convert_bbox_coco2yolo(img_width, img_height, bbox):
    """
    Convert bounding box from COCO  format to YOLO format

    Parameters
    ----------
    img_width : int
        width of image
    img_height : int
        height of image
    bbox : list[int]
        bounding box annotation in COCO format:
        [top left x position, top left y position, width, height]

    Returns
    -------
    list[float]
        bounding box annotation in YOLO format:
        [x_center_rel, y_center_rel, width_rel, height_rel]
    """

    # YOLO bounding box format: [x_center, y_center, width, height]
    # (float values relative to width and height of image)
    x_tl, y_tl, w, h = bbox

    dw = 1.0 / img_width
    dh = 1.0 / img_height

    x_center = x_tl + w / 2.0
    y_center = y_tl + h / 2.0

    x = x_center * dw
    y = y_center * dh
    w = w * dw
    h = h * dh

    return [x, y, w, h]


def convert_coco_json_to_yolo_txt(output_path, json_file):

    path = make_folders(output_path)

    with open(json_file) as f:
        json_data = json.load(f)

    # write _darknet.labels, which holds names of all classes (one class per line)
    label_file = os.path.join(output_path, "_darknet.labels")
    with open(label_file, "w") as f:
        for category in tqdm(json_data["categories"], desc="Categories"):
            category_name = category["name"]
            f.write(f"{category_name}\n")

    for image in tqdm(json_data["images"], desc="Annotation txt for each iamge"):
        img_id = image["id"]
        img_name = image["file_name"]
        img_width = image["width"]
        img_height = image["height"]

        anno_in_image = [anno for anno in json_data["annotations"] if anno["image_id"] == img_id]
        anno_txt = os.path.join(output_path, img_name.split("/")[-1].split(".")[0] + ".txt")
        with open(anno_txt, "w") as f:
            for anno in anno_in_image:
                category = anno["category_id"]
                print(category)
                bbox_COCO = anno["bbox"]
                x, y, w, h = convert_bbox_coco2yolo(img_width, img_height, bbox_COCO)
                f.write(f"{category} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    print("Converting COCO Json to YOLO txt finished!")


def draw_bounding_boxes(image_path, bbox_path):
    # Read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Get image dimensions
    height, width, _ = image.shape

    # Create a figure and axis
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Define a list of colors for different labels
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "cyan", "magenta", "lime", "brown", "navy"]

    # Read the bounding box file
    with open(bbox_path, "r") as file:
        for line in file:
            label, x_center, y_center, box_width, box_height = map(float, line.split())

            # Convert YOLO format (relative) to bounding box (absolute) coordinates
            x_center *= width
            y_center *= height
            box_width *= width
            box_height *= height

            # Calculate the top-left corner of the bounding box
            x_min = x_center - box_width / 2
            y_min = y_center - box_height / 2

            # Assign a color to the label
            color = colors[int(label) % len(colors)]

            # Create a rectangle patch
            rect = patches.Rectangle(
                (x_min, y_min), box_width, box_height, linewidth=1, edgecolor=color, facecolor="none"
            )

            # Add the rectangle to the plot
            ax.add_patch(rect)

            # Add the label text above the rectangle
            plt.text(x_min, y_min - 10, f"{int(label)}", color=color, fontsize=12, weight="bold")

    plt.axis("off")
    plt.show()


# Define the image and bbox file paths
image_file = "/home/michal/Desktop/IntelligentVirtualization/FLIR_ADAS_v2/images_thermal_train/images/video-nodg7n2aFzCWFJBnv-frame-002782-63QTzcSMgEJdqfg6N.jpg"
bbox_file = "/home/michal/Desktop/IntelligentVirtualization/FLIR_ADAS_v2/images_thermal_train/labels/video-nodg7n2aFzCWFJBnv-frame-002782-63QTzcSMgEJdqfg6N.txt"

# Draw bounding boxes on the image
draw_bounding_boxes(image_file, bbox_file)

# convert_coco_json_to_yolo_txt(
#     "/home/michal/Desktop/IntelligentVirtualization/FLIR_ADAS_v2/images_thermal_val/labels",
#     "/home/michal/Desktop/IntelligentVirtualization/FLIR_ADAS_v2/images_thermal_val/coco.json",
# )


# convert_coco_json_to_yolo_txt(
#     "/home/michal/Desktop/IntelligentVirtualization/FLIR_ADAS_v2/video_thermal_test/labels",
#     "/home/michal/Desktop/IntelligentVirtualization/FLIR_ADAS_v2/video_thermal_test/coco.json",
# )
