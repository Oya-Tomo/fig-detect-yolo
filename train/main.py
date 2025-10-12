import os
import subprocess

import wandb
from ultralytics import YOLO


def train():
    subprocess.run(["yolo", "settings", "wandb=True"])

    if os.getenv("WANDB_API_KEY") is not None:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
    else:
        print("WANDB_API_KEY not found in environment variables. Skipping wandb login.")
        raise EnvironmentError("WANDB_API_KEY not found in environment variables.")

    model = YOLO("yolov12l-doclaynet.pt")
    data_path = "dataset/data.yaml"
    model.train(
        data=data_path,
        epochs=300,
        imgsz=640,
        batch=64,
        lr0=0.001,
        name="yolov12l-doclaynet-fig-detect",
        project=f"{os.getcwd()}/runs",
    )
    model.save("yolov12l-doclaynet-fig-detect.pt")


if __name__ == "__main__":
    train()
