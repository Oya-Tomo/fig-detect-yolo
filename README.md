# Paper Figure Detection for Extraction

# Methods

## Auto Annotation

- base model: [yolo-doclaynet](https://huggingface.co/hantian/yolo-doclaynet)

### Figure & Caption Pair Annotation

Calculate the combination that minimizes the total distance between the caption and the figure.
`scipy.optimize.linear_sum_assignment` is used for this combination calculation.