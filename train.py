"""Example code for the DSC 140B SoCalGuessr project.

This example code trains a simple logistic regression model on the training images. It
is not meant to be a strong baseline -- it's just an example that shows you how to load
the data and how to export the model's weights.

This file is paired with `predict.py`, which loads the saved model and uses it to make
predictions on the test set.

Your training code does not *need* to look like this file, and you are allowed to use
any model and training procedure you'd like. However, you *can* use this file as a
starting point if you want.

"""

import pathlib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
# from tqdm import trange


# configuration ------------------------------------------------------------------------

# the path to the directory containing the training data
TRAIN_DIR = pathlib.Path("./data")

# the six classes of cities in our dataset
CLASSES = sorted(
    [
        "Anaheim",
        "Bakersfield",
        "Los_Angeles",
        "Riverside",
        "SLO",
        "San_Diego",
    ]
)

# we will later use this to convert a string class (like "Los_Angeles") to a numerical
# label (like 2), since PyTorch models work with numerical labels rather than strings.
CLASS_TO_NUMBER = {name: i for i, name in enumerate(CLASSES)}

# we'll resize all images to this size before feeding them into the model smaller image
# sizes means fewer parameters and therefore faster training, but also less information
# for the model to learn from
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 64

BATCH_SIZE = 64
LEARNING_RATE = 4e-4
EPOCHS = 20

# the percentage of the training data to set aside as a validation set.
VALIDATION_FRACTION = 0.2


# dataset ------------------------------------------------------------------------------

# this is a custom dataset loader that loads the images from disk and applies any
# transformations we specify.

class SoCalDataset(Dataset):
    """Loads images from the training set."""

    def __init__(self, root, transform=None):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.samples = []  # list of (path, label_index)

        # this determines the class label of an image based on its filename. For
        # example, the image "Los_Angeles-123.jpg" would be labeled as "Los_Angeles". We
        # then convert that string to a numerical label using the CLASS_TO_INDEX
        # dictionary defined above.
        for path in sorted(self.root.glob("*.jpg")):
            label = path.name.rsplit("-", 1)[0]
            self.samples.append((path, CLASS_TO_NUMBER[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# model --------------------------------------------------------------------------------

# this is a simple logistic regression model. It consists of a single linear layer that
# takes in the flattened pixel values and outputs a vector of class scores.

# when you're defining your own model, this is the part of the code that you'll likely
# change the most

class ColemanCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        C1, K1, P1 = 32,  3, 1
        C2, K2, P2 = 64,  3, 1
        C3, K3, P3 = 128, 3, 1
        self.conv1 = nn.Conv2d(3,  C1, kernel_size=K1, padding=P1)
        self.bn1   = nn.BatchNorm2d(C1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(C1, C2, kernel_size=K2, padding=P2)
        self.bn2   = nn.BatchNorm2d(C2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(C2, C3, kernel_size=K3, padding=P3)
        self.bn3   = nn.BatchNorm2d(C3)
        self.pool3 = nn.MaxPool2d(2, 2)
        h1 = ((IMAGE_HEIGHT + 2*P1 - K1) + 1) // 2
        w1 = ((IMAGE_WIDTH  + 2*P1 - K1) + 1) // 2
        h2 = ((h1 + 2*P2 - K2) + 1) // 2
        w2 = ((w1 + 2*P2 - K2) + 1) // 2
        h3 = ((h2 + 2*P3 - K3) + 1) // 2
        w3 = ((w2 + 2*P3 - K3) + 1) // 2
        self.fc1     = nn.Linear(C3 * h3 * w3, 256)
        self.bn4     = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(self.bn1(torch.relu(self.conv1(x))))
        x = self.pool2(self.bn2(torch.relu(self.conv2(x))))
        x = self.pool3(self.bn3(torch.relu(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.bn4(torch.relu(self.fc1(x))))
        return self.fc2(x)

# class LogisticRegression(nn.Module):
#     """A single linear layer — logistic regression on flattened pixels."""

#     def __init__(self, input_dim, num_classes):
#         super().__init__()
#         self.flatten = nn.Flatten()
#         self.linear = nn.Linear(input_dim, num_classes)

#     def forward(self, x):
#         x = self.flatten(x)
#         return self.linear(x)


# training -----------------------------------------------------------------------------


def main():
    # Step 1) define transformations on the images. Here we resize the images and
    # convert them to tensors.
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
            transforms.ToTensor(),
        ]
    )

    # Step 2) split the data into a train set and a validation set, and create data
    # loaders for each. See DSC 80 for why doing this is important!
    full_dataset = SoCalDataset(TRAIN_DIR, transform=transform)
    val_size = int(len(full_dataset) * VALIDATION_FRACTION)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Step 3) define the model, loss function, and optimizer. The model is a simple
    # logistic regression model defined above. The loss function is cross-entropy loss,
    # which is commonly used for multi-class classification problems. The optimizer is
    # Adam.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")

    input_dim = 3 * IMAGE_WIDTH * IMAGE_HEIGHT  # channels x height x width
    model = ColemanCNN(input_dim, len(CLASSES)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Step 4) the training loop.

    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total

        # evaluate on the validation set
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += images.size(0)
        val_accuracy = val_correct / val_total
        model.train()

        print(
            f"Epoch {epoch + 1}/{EPOCHS}  "
            f"loss: {avg_loss:.4f}  "
            f"accuracy: {accuracy:.4f}  "
            f"val_accuracy: {val_accuracy:.4f}"
        )

    torch.save(model.state_dict(), "model.pt")
    print("Saved model to model.pt")

if __name__ == "__main__":
    main()
