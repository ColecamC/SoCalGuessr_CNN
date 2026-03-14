"""Example prediction code for the DSC 140B SoCalGuessr project.

This file is paired with `train.py`, which trains a model and saves its weights to
`model.pt`. This file loads the saved model and uses it to make predictions on the test
set.

The autograder will call the `predict` function defined below. You are free to change
the implementation, but the function signature must stay the same: it must accept a path
to a directory of test images and return a dictionary mapping each filename to a
predicted label as a string (e.g. "Los_Angeles").

"""

import pathlib

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


# these must match the values used during training in train.py
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

# when we trained the model, we resized all images to this size before feeding them into
# the model. We need to do the same thing here, since the model's weights were trained
# on images of this size.
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 64


def load_and_transform_image(path):
    """Load an image from disk and apply the same transforms used during training.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a .jpg image file.

    Returns
    -------
    torch.Tensor
        A tensor of shape (1, 3, IMAGE_WIDTH, IMAGE_HEIGHT) ready to be fed into
        the model.

    """
    image = Image.open(path).convert("RGB")
    pipeline = transforms.Compose(
        [
            transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
            transforms.ToTensor(),
        ]
    )
    return pipeline(image).unsqueeze(0)  # add batch dimension


# we must re-define the model architecture here in order to load the saved weights.

class ColemanCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        C1, K1, P1 = 32,  3, 1
        C2, K2, P2 = 64,  3, 1
        C3, K3, P3 = 128, 3, 1
        self.conv1 = nn.Conv2d(3,  C1, kernel_size=K1, padding=P1)
        self.bn1 = nn.BatchNorm2d(C1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(C1, C2, kernel_size=K2, padding=P2)
        self.bn2 = nn.BatchNorm2d(C2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(C2, C3, kernel_size=K3, padding=P3)
        self.bn3 = nn.BatchNorm2d(C3)
        self.pool3 = nn.MaxPool2d(2, 2)
        h1 = ((IMAGE_HEIGHT + 2*P1 - K1) + 1) // 2
        w1 = ((IMAGE_WIDTH  + 2*P1 - K1) + 1) // 2
        h2 = ((h1 + 2*P2 - K2) + 1) // 2
        w2 = ((w1 + 2*P2 - K2) + 1) // 2
        h3 = ((h2 + 2*P3 - K3) + 1) // 2
        w3 = ((w2 + 2*P3 - K3) + 1) // 2
        self.fc1 = nn.Linear(C3 * h3 * w3, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

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


def predict(test_dir):
    """Load the saved model and predict a label for every image in `test_dir`.

    Parameters
    ----------
    test_dir : pathlib.Path
        Path to a directory containing .jpg test images.

    Returns
    -------
    dict[str, str]
        A dictionary mapping each image filename (e.g. "00001.jpg") to a predicted
        class label (e.g. "Los_Angeles").

    """
    test_dir = pathlib.Path(test_dir)

    # Step 1) load the trained model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 3 * IMAGE_WIDTH * IMAGE_HEIGHT
    model = ColemanCNN(input_dim, len(CLASSES))
    model.load_state_dict(torch.load("model.pt", weights_only=True, map_location=device))
    model.to(device)
    model.eval()

    # Step 2) run prediction on every test image.
    predictions = {}
    with torch.no_grad():
        for path in sorted(test_dir.glob("*.jpg")):
            image = load_and_transform_image(path).to(device)
            output = model(image)
            predicted_index = output.argmax(dim=1).item()
            predictions[path.name] = CLASSES[predicted_index]

    return predictions


if __name__ == "__main__":
    preds = predict("./data")
    print("Predictions:")
    for filename, label in sorted(preds.items()):
        print(f"{filename}: {label}")
