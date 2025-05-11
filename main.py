import os
import shutil
import logging
import argparse

import annotator_utils

logging.basicConfig(encoding='utf-8', level=logging.DEBUG)
LOGGER = logging.getLogger("cone_point_annotator")

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

    def save(self, output_ds_dir: str):
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
        input_dato: YOLOInput = self.filtered_inputs[-1]
        # process the input
        # get bboxes
        # from the bboxes, extract the patches
        # for each patch, get annotation from user
        # save the annotations (3 points)
        # remove the dato from the input list


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


if __name__ == "__main__":
    main()
