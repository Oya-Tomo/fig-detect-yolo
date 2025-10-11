from ultralytics import YOLO


def train():
    model = YOLO("yolov12l-doclaynet.pt")
    data_path = "dataset/data.yaml"
    model.train(
        data=data_path,
        epochs=100,
        imgsz=800,
        batch=16,
        name="yolov12l-doclaynet-fig-detect",
    )
    model.save("yolov12l-doclaynet-fig-detect.pt")


if __name__ == "__main__":
    train()
