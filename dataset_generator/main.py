import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import yaml
from collector import (
    collect_arxiv_papers,
    download_pdf,
    generate_short_hash,
    get_pdf_page_images,
)
from PIL import Image
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from ultralytics.engine.results import Results


def rect_overlap(box1: list[int], box2: list[int]) -> bool:
    no_collision = (
        box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3]
    )
    return not no_collision


def rect_distance(
    box1: list[int],
    box2: list[int],
    x_diff_coef: float = 1.0,
    y_diff_coef: float = 1.0,
) -> float:
    x_distance = max(0, box2[0] - box1[2], box1[0] - box2[2]) * x_diff_coef
    y_distance = max(0, box2[1] - box1[3], box1[1] - box2[3]) * y_diff_coef
    return (x_distance**2 + y_distance**2) ** 0.5


def merge_rects(box1: list[int], box2: list[int]) -> list[int]:
    x_min = min(box1[0], box2[0])
    y_min = min(box1[1], box2[1])
    x_max = max(box1[2], box2[2])
    y_max = max(box1[3], box2[3])
    return [x_min, y_min, x_max, y_max]


def create_dataset_item(
    hash: str,
    page: int,
    image: Image.Image,
    usage: str,
    results: Results,
    eliminate_empty_ratio: float = 0.7,
):
    image_id = f"{hash}_{page}"

    annotated_boxes = []

    old_classes = {
        v: k
        for k, v in {
            0: "Caption",
            1: "Footnote",
            2: "Formula",
            3: "List-item",
            4: "Page-footer",
            5: "Page-header",
            6: "Picture",
            7: "Section-header",
            8: "Table",
            9: "Text",
            10: "Title",
        }.items()
    }

    new_classes = {
        v: k
        for k, v in {
            0: "Caption",
            1: "Picture",
            2: "Table",
            3: "Picture-caption-pair",
            4: "Table-caption-pair",
        }.items()
    }

    # Auto-annotate figure-caption pairs
    picture_boxes = [
        box for box in results.boxes if int(box.cls.item()) == old_classes["Picture"]
    ]
    table_boxes = [
        box for box in results.boxes if int(box.cls.item()) == old_classes["Table"]
    ]
    caption_boxes = [
        box for box in results.boxes if int(box.cls.item()) == old_classes["Caption"]
    ]
    insert_boxes = picture_boxes + table_boxes
    if len(insert_boxes) != len(caption_boxes):
        return  # Skip this image if the number of figures and captions do not match

    if len(insert_boxes) == 0 and eliminate_empty_ratio > random.random():
        return  # Skip this image if there is no annotation

    annotated_boxes.extend(
        [
            (new_classes["Caption"],) + tuple(box.xywhn[0].tolist())
            for box in caption_boxes
        ]
    )
    annotated_boxes.extend(
        [
            (new_classes["Picture"],) + tuple(box.xywhn[0].tolist())
            for box in picture_boxes
        ]
    )
    annotated_boxes.extend(
        [(new_classes["Table"],) + tuple(box.xywhn[0].tolist()) for box in table_boxes]
    )

    ins_cap_dists = np.empty(
        (len(insert_boxes), len(caption_boxes)), dtype=float
    )  # dists[ins_idx][cap_idx] = distance
    ins_cap_class = np.empty(
        (len(insert_boxes), len(caption_boxes)), dtype=int
    )  # class[ins_idx][cap_idx] = new_class_id

    for ins_idx in range(len(insert_boxes)):
        for cap_idx in range(len(caption_boxes)):
            ins_box = insert_boxes[ins_idx].xyxy[0].tolist()
            cap_box = caption_boxes[cap_idx].xyxy[0].tolist()
            dist = rect_distance(
                ins_box,
                cap_box,
                x_diff_coef=2.0,
                y_diff_coef=1.0,
            )
            ins_cap_dists[ins_idx][cap_idx] = dist
            ins_cap_class[ins_idx][cap_idx] = (
                new_classes["Picture-caption-pair"]
                if int(insert_boxes[ins_idx].cls.item()) == old_classes["Picture"]
                else new_classes["Table-caption-pair"]
            )

    ins_cap_row_idx, ins_cap_col_idx = linear_sum_assignment(ins_cap_dists)
    for ins_idx, cap_idx in zip(ins_cap_row_idx, ins_cap_col_idx):
        ins_box = insert_boxes[ins_idx].xyxyn[0].tolist()
        cap_box = caption_boxes[cap_idx].xyxyn[0].tolist()
        merged_box = merge_rects(ins_box, cap_box)
        x_center = (merged_box[0] + merged_box[2]) / 2
        y_center = (merged_box[1] + merged_box[3]) / 2
        width = merged_box[2] - merged_box[0]
        height = merged_box[3] - merged_box[1]
        annotated_boxes.append(
            (
                ins_cap_class[ins_idx][cap_idx],
                x_center,
                y_center,
                width,
                height,
            )
        )

    image.save(f"dataset/images/{usage}/{image_id}.png")
    with open(f"dataset/labels/{usage}/{image_id}.txt", "w") as f:
        for box in annotated_boxes:
            class_id, x_center, y_center, width, height = box
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


@dataclass
class DatasetConfig:
    dpi: int = 150
    batch_size: int = 15
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    max_results: int = 100
    eliminate_empty_ratio: float = 0.7


def main(config: DatasetConfig = DatasetConfig()):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    os.makedirs("dataset/papers", exist_ok=True)

    os.makedirs("dataset/images/train", exist_ok=True)
    os.makedirs("dataset/images/val", exist_ok=True)
    os.makedirs("dataset/images/test", exist_ok=True)

    os.makedirs("dataset/labels/train", exist_ok=True)
    os.makedirs("dataset/labels/val", exist_ok=True)
    os.makedirs("dataset/labels/test", exist_ok=True)

    with open("dataset/data.yaml", "w+") as data_yaml:
        yaml.dump(
            {
                "path": "./dataset",
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {
                    0: "Caption",
                    1: "Picture",
                    2: "Table",
                    3: "Picture-caption-pair",
                    4: "Table-caption-pair",
                },
            },
            data_yaml,
            sort_keys=False,
        )

    model = YOLO("yolov12l-doclaynet.pt")

    categories = [
        "cs.AI",
        "cs.AR",
        "cs.CC",
        "cs.CE",
        "cs.CG",
        "cs.CL",
        "cs.CR",
        "cs.CV",
        "cs.CY",
        "cs.DB",
        "cs.DC",
        "cs.DL",
        "cs.DM",
        "cs.DS",
        "cs.ET",
        "cs.FL",
        "cs.GL",
        "cs.GR",
        "cs.GT",
        "cs.HC",
        "cs.IR",
        "cs.IT",
        "cs.LG",
        "cs.LO",
        "cs.MA",
        "cs.MM",
        "cs.MS",
        "cs.NA",
        "cs.NE",
        "cs.NI",
        "cs.OH",
        "cs.OS",
        "cs.PF",
        "cs.PL",
        "cs.RO",
        "cs.SC",
        "cs.SD",
        "cs.SE",
        "cs.SI",
        "cs.SY",
        "physics.acc-ph",
        "physics.ao-ph",
        "physics.app-ph",
        "physics.atm-clus",
        "physics.atom-ph",
        "physics.bio-ph",
        "physics.chem-ph",
        "physics.class-ph",
        "physics.comp-ph",
        "physics.data-an",
        "physics.ed-ph",
        "physics.flu-dyn",
        "physics.gen-ph",
        "physics.geo-ph",
        "physics.hist-ph",
        "physics.ins-det",
        "physics.med-ph",
        "physics.optics",
        "physics.plasm-ph",
        "physics.pop-ph",
        "physics.soc-ph",
        "physics.space-ph",
        "q-bio.BM",
        "q-bio.CB",
        "q-bio.GN",
        "q-bio.MN",
        "q-bio.NC",
        "q-bio.OT",
        "q-bio.PE",
        "q-bio.QM",
        "q-bio.SC",
        "q-bio.TO",
        "eess.AS",
        "eess.IV",
        "eess.SP",
        "eess.SY",
    ]

    queries = [f"cat:{cat}" for cat in categories]
    id_list = None
    start = 0

    print(f"Starting downloading papers...")
    for query in queries:
        print(f"Query: {query}")
        papers = collect_arxiv_papers(
            search_query=query,
            id_list=id_list,
            start=start,
            max_results=config.max_results,
        )
        print(f"    Found {len(papers)} papers")
        for paper in papers:
            print(f"    {paper.id} - {paper.title}")
            paper_hash = generate_short_hash(paper.id)
            pdf_path = f"dataset/papers/{paper_hash}.pdf"
            if os.path.exists(pdf_path):
                print(f"        PDF already exists, skipping download")
                continue
            if not download_pdf(paper.pdf, pdf_path):
                print(f"        Failed to download PDF")

    pdf_paths = [
        f"dataset/papers/{file}"
        for file in os.listdir("dataset/papers")
        if file.endswith(".pdf")
    ]

    page_pool: list[
        tuple[
            str,
            int,
            Image.Image,
            str,
        ]
    ] = []  # contains (hash, page, image, usage)
    for pdf_idx, pdf_path in enumerate(pdf_paths):
        print(f"Processing {pdf_idx+1}/{len(pdf_paths)}: {pdf_path}")

        paper_hash = os.path.splitext(os.path.basename(pdf_path))[0]

        images = get_pdf_page_images(pdf_path, dpi=config.dpi)
        images_usage = random.choices(
            ["train", "val", "test"],
            weights=[config.train_split, config.val_split, config.test_split],
            k=len(images),
        )
        for image_idx, image in enumerate(images):
            page_pool.append((paper_hash, image_idx, image, images_usage[image_idx]))

        if len(page_pool) < config.batch_size:
            continue

        for _ in range(len(page_pool) // config.batch_size):
            batch = page_pool[: config.batch_size]
            page_pool = page_pool[config.batch_size :]

            paper_hashes, page_indices, images, images_usage = zip(*batch)
            images = list(images)
            images_usage = list(images_usage)
            images_predict = model.predict(images)

            for i in range(len(images)):
                create_dataset_item(
                    hash=paper_hashes[i],
                    page=page_indices[i],
                    image=images[i],
                    usage=images_usage[i],
                    results=images_predict[i],
                    eliminate_empty_ratio=config.eliminate_empty_ratio,
                )

            del images_predict
            torch.cuda.empty_cache()


if __name__ == "__main__":
    config = DatasetConfig(
        dpi=150,
        batch_size=100,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        max_results=200,
        eliminate_empty_ratio=0.95,
    )
    main(config)
