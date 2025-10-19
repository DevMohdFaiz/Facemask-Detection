import matplotlib.pyplot as plt
import os
from PIL import Image
from pathlib import Path
import random
import torch
from collections import Counter
import pandas as pd
from tqdm.notebook import tqdm
import torchvision.transforms as T

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_random_images(train_dir, classes, nrows, ncols):  
        """
        Plot random images from the dataset
        """
        fig, ax = plt.subplots(figsize=(10, 9), nrows=nrows, ncols=ncols)
        fig.suptitle('Sample Images')
        ax = ax.flatten()

        for idx, axis in enumerate(ax):
                random_path = Path(os.path.join(train_dir, random.choice(classes)))
                random_image = Path(os.path.join(random_path, random.choice(os.listdir(random_path))))
                img_tensor = torch.tensor(plt.imread(random_image))
                img_name = Image.open(random_image)
                axis.imshow(img_tensor)
                axis.set_xticks([])
                axis.set_yticks([])
                axis.set_title(f'{random_image.parent.stem}, {img_name.mode}', fontsize=9)
        


def class_counts(dataset):
    c = Counter(x[1] for x in tqdm(dataset))
    try:
        class_to_index = dataset.class_to_idx
    except AttributeError:
        class_to_index = dataset.dataset.class_to_idx
    return pd.Series({cat: c[idx] for cat, idx in class_to_index.items()})


def train_epoch(model, optimizer, loss_fn, data_loader, device="cpu"):
    training_loss = 0.0
    model=model.to(device)
    model.train()

    # Iterate over all batches in the training set to complete one epoch
    for inputs, targets in tqdm(data_loader, desc="Training", leave=False):
        optimizer.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)

        output = model(inputs)
        # print("Targets dtype:", targets.dtype, "shape:", targets.shape, "unique:", targets.unique())

        loss = loss_fn(output, targets)

        loss.backward()
        optimizer.step()
        training_loss += loss.data.item() * inputs.size(0)

    return training_loss / len(data_loader.dataset)


def predict(model, data_loader, device="cpu"):
    all_probs = torch.tensor([]).to(device)

    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Predicting", leave=False):
            inputs = inputs.to(device)
            output = model(inputs)
            probs = torch.nn.functional.softmax(output, dim=1)
            all_probs = torch.cat((all_probs, probs), dim=0)

    return all_probs


def score(model, data_loader, loss_fn, device="cpu"):
    total_loss = 0
    total_correct = 0

    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Scoring", leave=False):
            inputs = inputs.to(device)
            output = model(inputs)

            targets = targets.to(device)
            loss = loss_fn(output, targets)
            total_loss += loss.data.item() * inputs.size(0)

            correct = torch.eq(torch.argmax(output, dim=1), targets)
            total_correct += torch.sum(correct).item()

    # n_observations = data_loader.batch_size * len(data_loader)
    n_observations = len(data_loader.dataset)
    average_loss = total_loss / n_observations
    accuracy = total_correct / n_observations
    return average_loss, accuracy


def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device="cpu", use_train_accuracy=True):
    # Track the model progress over epochs
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(1, epochs + 1):
        # Train one epoch
        training_loss = train_epoch(model, optimizer, loss_fn, train_loader, device)

        # Evaluate training results
        if use_train_accuracy:
            train_loss, train_accuracy = score(model, train_loader, loss_fn, device)
        else:
            train_loss = training_loss
            train_accuracy = 0
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Test on validation set
        validation_loss, validation_accuracy = score(model, val_loader, loss_fn, device)
        val_losses.append(validation_loss)
        val_accuracies.append(validation_accuracy)

        print(f"Epoch: {epoch}")
        print(f"    Training loss: {train_loss:.2f}")
        if use_train_accuracy:
            print(f"    Training accuracy: {train_accuracy:.2f}")
        print(f"    Validation loss: {validation_loss:.2f}")
        print(f"    Validation accuracy: {validation_accuracy:.2f}")

    return train_losses, val_losses, train_accuracies, val_accuracies



def unnormalize_image(img, mean= [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        mean = torch.tensor(mean).view(1, 1, -1)
        std = torch.tensor(std).view(1, 1, -1)
        return img.permute(1, 2, 0) * std + mean


def make_prediction(model, dataset):
    correct = 0
    random_idx = random.choice(range(len(dataset)))
    # preds, actual_labels, all_images = [], [], []
    random_img, random_label = dataset[random_idx]
    random_img = random_img.to(device)
    with torch.no_grad():
        output = model(random_img.unsqueeze(0))
        pred = output.argmax().item()
        predicted_label = class_names[pred]
        actual_label = class_names[random_label]
    
    
        
    unnormalized_img = unnormalize_image(random_img)
    
    fig, ax = plt.subplots(figsize=(3,3))
    fig.tight_layout()
    ax.imshow(unnormalized_img)
    color = 'green' if predicted_label == actual_label else 'red'
    ax.set_title(f'Actual: {actual_label}\n Pred:{predicted_label}', fontsize=12, color=color)
    ax.axis('off')


def run_prediction(img_path, model):
        user_img = train_transforms(Image.open(img_path).convert('RGB'))
        pred = torch.argmax(model(user_img.unsqueeze(0))).item()
        prediction = class_names[pred]
        unnormalized_user_img= unnormalize_image(user_img)
        fig, ax = plt.subplots()
        ax.imshow(unnormalized_user_img)
        ax.axis('off')
        ax.set_title(f'Predicted: {prediction}')
        fig.show()
        return prediction, class_names_dict[prediction];

