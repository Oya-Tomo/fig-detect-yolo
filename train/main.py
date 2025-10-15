import os
import subprocess

import wandb
from ultralytics import YOLO, settings


def train():
    settings.update({"wandb": True})

    if os.getenv("WANDB_API_KEY") is not None:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
    else:
        print("WANDB_API_KEY not found in environment variables. Skipping wandb login.")
        raise EnvironmentError("WANDB_API_KEY not found in environment variables.")

    model = YOLO("yolo12m.yaml")
    data_path = "dataset/data.yaml"
    model.train(
        data=data_path,
        epochs=300,
        imgsz=640,
        batch=96,
        optimizer="AdamW",
        lr0=0.0001,
        amp=True,
        name="yolo12m",
        project=f"fig-detect-yolo",
    )
    model.save("yolo12m-fig-detect.pt")


if __name__ == "__main__":
    train()
