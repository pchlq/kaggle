import os
import ast
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from model_dispatcher import MODEL_DISPATCHER
from dataset import BengaliDatasetTrain
from earlystopping import EarlyStopping
import sklearn.metrics

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #"cuda"
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
BASE_MODEL = os.environ.get("BASE_MODEL")


def macro_recall(pred_y, y, n_grapheme=168, n_vowel=11, n_consonant=7):
    
    pred_y = torch.split(pred_y, [n_grapheme, n_vowel, n_consonant], dim=1)
    pred_labels = [torch.argmax(py, dim=1).cpu().numpy() for py in pred_y]

    y = y.cpu().numpy()

    recall_grapheme = sklearn.metrics.recall_score(pred_labels[0], y[:, 0], average='macro', zero_division=1)
    recall_vowel = sklearn.metrics.recall_score(pred_labels[1], y[:, 1], average='macro', zero_division=1)
    recall_consonant = sklearn.metrics.recall_score(pred_labels[2], y[:, 2], average='macro', zero_division=1)
    scores = [recall_grapheme, recall_vowel, recall_consonant]
    final_score = np.average(scores, weights=[2, 1, 1])
    print(f'recall: grapheme {recall_grapheme}, vowel {recall_vowel}, consonant {recall_consonant}, 'f'total {final_score}, y {y.shape}')  
    return final_score


def focal_fn(outputs, targets, alpha=1, gamma=2):
    ce_loss = nn.functional.cross_entropy(outputs, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = (alpha * (1-pt)**gamma * ce_loss).mean()
    return focal_loss


def loss_fn(outputs, targets):
    o1, o2, o3 = outputs
    t1, t2, t3 = targets
    l1 = focal_fn(o1, t1) 
    l2 = focal_fn(o2, t2) 
    l3 = focal_fn(o3, t3)

    # l1 = nn.CrossEntropyLoss()(o1, t1) # grapheme_root 0.9
    # l2 = nn.CrossEntropyLoss()(o2, t2) # vowel_diacritic 0.06
    # l3 = nn.CrossEntropyLoss()(o3, t3) # consonant_diacritic 0.04
    
    return (l1 + l2 + l3) / 3


def train(dataset, data_loader, model, optimizer):
    model.train()
    final_loss = 0
    counter = 0
    final_outputs = []
    final_targets = []

    for bi, d in tqdm(enumerate(data_loader), total=int(len(dataset) / data_loader.batch_size)):
        counter += 1
        image = d["image"]
        grapheme_root = d["grapheme_root"]
        vowel_diacritic = d["vowel_diacritic"]
        consonant_diacritic = d["consonant_diacritic"]
        
        image = image.to(DEVICE, dtype=torch.float)
        grapheme_root = grapheme_root.to(DEVICE, dtype=torch.long)
        vowel_diacritic = vowel_diacritic.to(DEVICE, dtype=torch.long)
        consonant_diacritic = consonant_diacritic.to(DEVICE, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(image)
        targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
        loss = loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()

        final_loss += loss

        o1, o2, o3 = outputs
        t1, t2, t3 = targets
        final_outputs.append(torch.cat((o1,o2,o3), dim=1))
        final_targets.append(torch.stack((t1,t2,t3), dim=1))

        #if bi % 10 == 0:
        #    break
    final_outputs = torch.cat(final_outputs)
    final_targets = torch.cat(final_targets)

    print("=================Train=================")
    macro_recall_score = macro_recall(final_outputs, final_targets)
    return final_loss/counter , macro_recall_score



def evaluate(dataset, data_loader, model):
    with torch.no_grad():
        model.eval()
        final_loss = 0
        counter = 0
        final_outputs = []
        final_targets = []
        for bi, d in tqdm(enumerate(data_loader), total=int(len(dataset) / data_loader.batch_size)):
            counter += 1
            image = d["image"]
            grapheme_root = d["grapheme_root"]
            vowel_diacritic = d["vowel_diacritic"]
            consonant_diacritic = d["consonant_diacritic"]
            
            image = image.to(DEVICE, dtype=torch.float)
            grapheme_root = grapheme_root.to(DEVICE, dtype=torch.long)
            vowel_diacritic = vowel_diacritic.to(DEVICE, dtype=torch.long)
            consonant_diacritic = consonant_diacritic.to(DEVICE, dtype=torch.long)

            outputs = model(image)
            targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
            loss = loss_fn(outputs, targets)
            final_loss += loss

            o1, o2, o3 = outputs
            t1, t2, t3 = targets
            
            final_outputs.append(torch.cat((o1,o2,o3), dim=1))
            final_targets.append(torch.stack((t1,t2,t3), dim=1))
        
        final_outputs = torch.cat(final_outputs)
        final_targets = torch.cat(final_targets)

        print("=================Test=================")
        macro_recall_score = macro_recall(final_outputs, final_targets)

    return final_loss/counter , macro_recall_score


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
        num_workers=5
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
        num_workers=5
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", 
                                            patience=5, factor=0.3, min_lr=1e-10, verbose=True)

    # early_stopping = EarlyStopping(patience=5, verbose=True)

    best_score = -1
    print("FOLD : ", VALIDATION_FOLDS[0] )

    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)

    for epoch in range(1, EPOCHS+1):

        train_loss, train_score = train(train_dataset,train_loader, model, optimizer)
        val_loss, val_score = evaluate(valid_dataset, valid_loader, model)
        scheduler.step(val_loss)

        # if val_score > best_score:
        #     best_score = val_score
            # torch.save(model.state_dict(), f"{BASE_MODEL}_fold{VALIDATION_FOLDS[0]}.bin")
        torch.save(model.state_dict(), f"{BASE_MODEL}_fold{VALIDATION_FOLDS[0]}.bin")
        epoch_len = len(str(EPOCHS))
        print_msg = (f'[{epoch:>{epoch_len}}/{EPOCHS:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'train_score: {train_score:.5f} ' +
                     f'valid_loss: {val_loss:.5f} ' +
                     f'valid_score: {val_score:.5f}'
                    )
        
        print(print_msg)

        # early_stopping(val_score, model)
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break

    # for epoch in range(EPOCHS):
    #     train(train_dataset, train_loader, model, optimizer)
    #     val_score = evaluate(valid_dataset, valid_loader, model)
    #     print(f"Eval -> epoch: {epoch}, final loss: {val_score}")
    #     scheduler.step(val_score)
    #     torch.save(model.state_dict(), f"{BASE_MODEL}_fold{VALIDATION_FOLDS[0]}.bin")

if __name__ == "__main__":
    main()