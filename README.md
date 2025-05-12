# Cone Point Annotator
This is a simple tool written for a very specific usecase; adding keypoints to Formula student cone bounding box datasets.
The idea is that you input a YOLO detection dataset and via annotation, you create a YOLO pose dataset with keypoints.

## Installation
First, clone the repository and install the `uv` package manager for Python.
Then, the dependencies should be automatically installed on the first run.

## Usage
The tool is meant to run on standard COCO-like datasets. These usually have a main directory; in our example, let's say
`dataset`. In this directory, you have several subsections, like `dataset/train` and `dataset/val`. In each of these subfolders,
there are two folders: `dataset/train/labels` and `dataset/train/images`, containing labels and images respectively. The labels
are in the format of `class_id cx cy width height` for each bounding box, where each of the spatial values are a fraction of the image size.

Our tool takes in as input the original bbox dataset and outputs a new one, the YOLO pose dataset. This dataset contains the same folders and the same images, just the annotations are slightly different; each line looks like:

```
class_id cx cy width height topx topy leftx lefty rightx righty
```

You run the tool by running `uv run main.py --input_dataset [BBOX_DATASET_FOLDER] --output_dataset [DESIRED_OUTPUT_FOLDER]`. For each of the input dataset samples, it takes the bounding boxes, cuts out the cones inside the bounding boxes, and then asks you to select the three points on the cone in the order of *top*, *left*, *right*. While annotating one sample, there is no state kept on the disk; if you end the program there and then, the annotations for that given image will not be saved. Once you finish annotating it though, the program will see it when you launch with the same output folder next time and will not ask you to re-annotate.

## Bugreporting
This was coded up in like 2h; please let me know if there are any bugs that you need fixed.
