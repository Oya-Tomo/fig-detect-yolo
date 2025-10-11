import argparse

from ultralytics.data.utils import visualize_image_annotations

parser = argparse.ArgumentParser()
parser.add_argument(
    "--img_path",
    "-i",
    type=str,
    default="dataset/images/test/0.png",
)
parser.add_argument(
    "--label_path",
    "-l",
    type=str,
    default="dataset/labels/test/0.txt",
)
args = parser.parse_args()

label_map = {
    0: "Caption",
    1: "Picture",
    2: "Table",
    3: "Picture-caption-pair",
    4: "Table-caption-pair",
}


visualize_image_annotations(
    args.img_path,
    args.label_path,
    label_map=label_map,
)
