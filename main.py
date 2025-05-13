import os
import shutil
import logging
import argparse

import cv2

import annotator_utils

logging.basicConfig(encoding='utf-8', level=logging.DEBUG)
LOGGER = logging.getLogger("cone_point_annotator")

def visualize_and_get_points(image_path, label_path, resized_shape=(256, 256)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")

    orig_h, orig_w = image.shape[:2]

    # Load YOLO labels
    with open(label_path, 'r') as f:
        lines = f.readlines()

    datalines = []
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        class_id, cx, cy, w, h = map(float, parts)


        # Convert normalized to absolute coordinates
        cx *= orig_w
        cy *= orig_h
        w *= orig_w
        h *= orig_h

        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)

        bbox_pixel_buffer = 15
        # Clip coordinates
        x1 = max(0, x1 - bbox_pixel_buffer)
        y1 = max(0, y1 - bbox_pixel_buffer)
        x2 = min(orig_w, x2 + bbox_pixel_buffer)
        y2 = min(orig_h, y2 + bbox_pixel_buffer)

        crop = image[y1:y2, x1:x2]
        crop_resized = cv2.resize(crop, resized_shape)

        points = []

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 3:
                points.append((x, y))
                labels = ['top', 'left', 'right']
                cv2.circle(crop_resized, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(crop_resized, labels[len(points) - 1], (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("Crop", crop_resized)

        print(f"\nShowing crop #{i+1} (class {int(class_id)}). Click: top, left, right.")
        cv2.imshow("Crop", crop_resized)
        cv2.setMouseCallback("Crop", click_event)
        while len(points) < 3:
            cv2.waitKey(1)
        cv2.destroyWindow("Crop")

        # Map resized points back to original image coordinates
        scale_x = (x2 - x1) / resized_shape[0]
        scale_y = (y2 - y1) / resized_shape[1]
        original_points = [((x * scale_x + x1) / orig_w, (y * scale_y + y1) / orig_h) for (x, y) in points]
        class_id, cx, cy, w, h = map(float, parts)
        dataline = PoseDataLine(class_id, cx, cy, w, h, original_points[0], original_points[1], original_points[2])
        datalines.append(dataline)
    return datalines


class PoseDataLine:
    def __init__(self, class_id, cx, cy, w, h, top, left, right):
        self.class_id = int(class_id)
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.top = top
        self.left = left
        self.right = right

    def serialize(self):
        return f"{self.class_id} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f} " \
               f"{self.top[0]:.6f} {self.top[1]:.6f} " \
               f"{self.left[0]:.6f} {self.left[1]:.6f} " \
               f"{self.right[0]:.6f} {self.right[1]:.6f}"


class YOLOInput:
    def __init__(self, image_path, label_path):
        self.image_path = image_path
        self.label_path = label_path
        self.subfolder = annotator_utils.get_ds_subfolder(image_path)

    def check_outputs_exist(self, output_ds_dir: str):
        image_file = os.path.basename(self.image_path)
        label_file = os.path.basename(self.label_path)
        image_output_path = output_ds_dir + "/images/" + image_file
        label_output_path = output_ds_dir + "/labels/" + label_file
        return os.path.isfile(image_output_path) or os.path.isfile(label_output_path)

    def save(self, output_ds_dir: str, datalines):
        os.makedirs(output_ds_dir + "/images", exist_ok=True)
        os.makedirs(output_ds_dir + "/labels", exist_ok=True)
        image_file = os.path.basename(self.image_path)
        label_file = os.path.basename(self.label_path)
        image_output_path = output_ds_dir + "/images/" + image_file
        label_output_path = output_ds_dir + "/labels/" + label_file
        # TODO save the image
        LOGGER.info(f"Saving image at path {image_output_path}")
        # TODO serialize and save the bboxes and keypoints
        LOGGER.info(f"Saving label at path {label_output_path}")
        serialized = list(map(lambda x: x.serialize(), datalines))
        full = "\n".join(serialized)

        with open(label_output_path, "w") as f:
            f.write(full)

class Annotator:
    def __init__(self, input_folder: str, output_folder: str, subfolders: list[str]):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.subfolders = subfolders
        self.input = []
        self.filtered_inputs = []

    def index_input(self):
        self.inputs = []
        for subfolder in self.subfolders:
            folder_path = self.input_folder + "/" + subfolder
            images = folder_path + "/images"
            labels = folder_path + "/labels"
            image_paths = [os.path.join(images, f) for f in os.listdir(images) if os.path.isfile(os.path.join(images, f))]
            label_paths = [os.path.join(labels, f) for f in os.listdir(labels) if os.path.isfile(os.path.join(labels, f))]
            matched = annotator_utils.match_images_labels(image_paths, label_paths)
            for img_path, label_path in matched:
                yolo_input = YOLOInput(img_path, label_path)
                self.inputs.append(yolo_input)

        self.filtered_inputs = list(filter(lambda data: not data.check_outputs_exist(self.output_folder), self.inputs))

    def process_one(self):
        if not self.filtered_inputs:
            return False
        input_dato: YOLOInput = self.filtered_inputs[-1]
        datalines = visualize_and_get_points(input_dato.image_path, input_dato.label_path)
        input_dato.save(self.output_folder, datalines)
        return True


def main():
    parser = argparse.ArgumentParser(
        prog="Cone point annotator",
    )
    parser.add_argument("--input_dataset", type=str, required=True, help="The input YOLO coco-like dataset")
    parser.add_argument("--output_dataset", type=str, required=True, help="The path where you want to put your YOLO pose dataset")
    args = parser.parse_args()
    subfolders = list(filter(lambda path: os.path.isdir(args.input_dataset + "/" + path), os.listdir(args.input_dataset)))
    annotator = Annotator(args.input_dataset, args.output_dataset, subfolders)
    annotator.index_input()
    while True:
        retval = annotator.process_one()
        if not retval:
            LOGGER.info("All data processed!")
            break

if __name__ == "__main__":
    main()
