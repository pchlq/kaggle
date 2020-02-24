import os
import ast
import torch
import torch.nn as nn
from model_dispatcher import MODEL_DISPATCHER
from dataset import BengaliDatasetTrain

DEVICE = "cuda"
TRAINING_FOLDS_CSV = os.environ.get("TRAINING_FOLDS_CSV")
IMG_HEIGHT = int(os.environ.get("IMG_HEIGHT"))
IMG_WIDTH = int(os.environ.get("IMG_WIDTH"))
EPOCHS = int(os.environ.get("EPOCHS"))

TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE"))
TEST_BATCH_SIZE = int(os.environ.get("TEST_BATCH_SIZE"))

MODEL_MEAN = ast.literal_eval(os.environ.get("MODEL_MEAN"))
MODEL_STD = ast.literal_eval(os.environ.get("MODEL_STD"))

TRAINING_FOLDS = ast.literal_eval(os.environ.get("TRAINING_FOLDS"))
VALIDATION_FOLDS = ast.literal_eval(os.environ.get("VALIDATION_FOLDS"))
BASE_MODEL = int(os.environ.get("BASE_MODEL"))

def main():
    model = MODEL_DISPATCHER[BASE_MODEL](pretrained=True)
    model.to(DEVICE)

    # for training set
    train_dataset = BengaliDatasetTrain(
        folds=TRAINING_FOLDS,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        mean=MODEL_MEAN,
        std=MODEL_STD
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    # for validation set
    valid_dataset = BengaliDatasetTrain(
        folds=VALIDATION_FOLDS,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        mean=MODEL_MEAN,
        std=MODEL_STD
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", 
                                            patience=5, factor=0.3, verbose=True)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel

    for epoch in range(EPOCHS):
        train()
        val_score = evaluate()
        scheduler.step(val_score)
        torch.save(model.state_dict(), f"{BASE_MODEL}_fold{VALIDATION_FOLDS[0]}.bin")

if __init__ == "__main__":
    main()