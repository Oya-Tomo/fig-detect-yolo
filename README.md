# Paper Figure Detection for Extraction

# Methods

## Auto Annotation

- base model: [yolo-doclaynet](https://huggingface.co/hantian/yolo-doclaynet)

### Figure & Caption Pair Annotation

Calculate the combination that minimizes the total distance between the caption and the figure.
`scipy.optimize.linear_sum_assignment` is used for this combination calculation.

### Excluding irrelevant labels

Filter out irrelevant class labels to improve detection accuracy

- Exclude labels below
    - Footnote
    - Formula
    - List-item
    - Page-footer
    - Page-header
    - Section-header
    - Text
    - Title

## Fine-tuning

- base model: [yolo-doclaynet](https://huggingface.co/hantian/yolo-doclaynet)

- detection classes
    - 0: Caption
    - 1: Picture
    - 2: Table
    - 3: Picture-caption-pair
    - 4: Table-caption-pair


