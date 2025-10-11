from ultralytics import YOLO


def train():
    model = YOLO("yolov12l-doclaynet.pt")
    data_path = "dataset/data.yaml"
    model.train(
        data=data_path,
        epochs=100,
        imgsz=640,
        batch=64,
        name="yolov12l-doclaynet-fig-detect",
    )
    model.save("yolov12l-doclaynet-fig-detect.pt")


if __name__ == "__main__":
    train()
