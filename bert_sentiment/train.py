import os
import torch
from loguru import logger
from pytorch_transformers import BertConfig, BertForSequenceClassification
from tqdm import tqdm

from .data import SSTDataset, TweetDataset

##original code mostly but added comments and changed which data is used at what point

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, lossfn, optimizer, dataset, batch_size=32):
    generator = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    model.train()
    train_loss, train_acc = 0.0, 0.0
    for batch, labels in tqdm(generator):
        batch, labels = batch.to(device), labels.to(device)
        optimizer.zero_grad()
        loss, logits = model(batch, labels=labels) #pass data through network
        err = lossfn(logits, labels)
        loss.backward() #backpropagation
        optimizer.step() #taking a step before going to the next training example

        train_loss += loss.item()
        pred_labels = torch.argmax(logits, axis=1) # takes the label with the highest probability
        train_acc += (pred_labels == labels).sum().item()
    train_loss /= len(dataset)
    train_acc /= len(dataset)
    return train_loss, train_acc


def evaluate_one_epoch(model, lossfn, optimizer, dataset, batch_size=32): #optimizer not needed in non-traning phase
    generator = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    model.eval()
    loss, acc = 0.0, 0.0
    with torch.no_grad():
        for batch, labels in tqdm(generator):
            batch, labels = batch.to(device), labels.to(device)
            logits = model(batch)[0]
            error = lossfn(logits, labels) #pass data through network
            loss += error.item()
            pred_labels = torch.argmax(logits, axis=1) # get label with highest prob
            acc += (pred_labels == labels).sum().item()
    loss /= len(dataset)
    acc /= len(dataset)
    return loss, acc


def train(
    root=True,
    binary=False,
    bert="bert-large-uncased",
    epochs=30,
    batch_size=32,
    save=False,
):
    trainset = SSTDataset("train", root=root, binary=binary) #loading and tokenizing the data
    devset = SSTDataset("dev", root=root, binary=binary)
    testset = SSTDataset("test", root=root, binary=binary)
    
    ##my code:
    if binary:
        other_testset = TweetDataset()
    ##

    config = BertConfig.from_pretrained(bert)
    if not binary:
        config.num_labels = 5
    model = BertForSequenceClassification.from_pretrained(bert, config=config)

    model = model.to(device)
    lossfn = torch.nn.CrossEntropyLoss() #loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) #adam = kind of stochastic gradient descent method + optimizer

    for epoch in range(1, epochs):
        train_loss, train_acc = train_one_epoch(
            model, lossfn, optimizer, trainset, batch_size=batch_size
        )
        val_loss, val_acc = evaluate_one_epoch(
            model, lossfn, optimizer, devset, batch_size=batch_size
        )

        logger.info(f"epoch={epoch}")
        logger.info(
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
        )
        logger.info(
            f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}"
        )

        if epoch == 29: #I added this if statement
            test_loss, test_acc = evaluate_one_epoch(
            model, lossfn, optimizer, testset, batch_size=batch_size
            )

            logger.info(
                f"test_loss={test_loss:.4f}, test_acc={test_acc:.3f}"
            )
        ## my code
            if binary:
                other_test_loss, other_test_acc = evaluate_one_epoch(
                model, lossfn, optimizer, other_testset, batch_size=batch_size
                )

                logger.info(
                f"other_test_loss={other_test_loss:.4f}, other_test_acc={other_test_acc:.3f}"
                )
        ## 
      
        if save:
            label = "binary" if binary else "fine"
            nodes = "root" if root else "all"
            torch.save(model, f"{bert}__{nodes}__{label}__e{epoch}.pickle")

    logger.success("Done!")
