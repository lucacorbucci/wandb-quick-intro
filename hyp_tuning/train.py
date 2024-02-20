import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import wandb
import argparse


# Define the model we will use
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


# Define two functions, one for training the model and one for testing it
def train_model(model, train_loader, device, optimizer):
    """Just the simple training function taken from the pytorch
    tutorial.
    """
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(
                "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                    epoch + 1, num_epochs, i + 1, total_step, loss.item()
                )
            )

        wandb.log({"loss": loss, "step": i * epoch})


def test_model(model, test_loader, device):
    """Just the simple testing function taken from the pytorch
    tutorial."""
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


# To run the hyperparameter tuning via CLI we need to define the hyperparameters
# allowing to change them from the command line. We will use argparse

parser = argparse.ArgumentParser(description="Hyperparameter tuning example with Wandb")
parser.add_argument("--num_epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--optimizer", type=str, default="Adam")


args = parser.parse_args()

# _____________________________________________________________________________
# _____________________________________________________________________________

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyper parameters
num_epochs = args.num_epochs
num_classes = 10
batch_size = args.batch_size
learning_rate = args.learning_rate

# These two additional parameters are for logging to wandb
# the images and the corresponding model predictions.
NUM_BATCHES_TO_LOG = 10
NUM_IMAGES_PER_BATCH = 32

# _____________________________________________________________________________
# _____________________________________________________________________________

# Prepare the data
# MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root="../data/", train=True, transform=transforms.ToTensor(), download=True
)

# split the train into train and validation
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [55000, 5000])

test_dataset = torchvision.datasets.MNIST(
    root="../data/", train=False, transform=transforms.ToTensor()
)

# Data loader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)

val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset, batch_size=batch_size, shuffle=False
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False
)


# _____________________________________________________________________________
# _____________________________________________________________________________
# Train the model

model = ConvNet(num_classes).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
if args.optimizer == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
elif args.optimizer == "SGD":
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Initialize wandb
wandb_run = wandb.init(
    project="wandb-quick-intro",  # name of the project in which we want to store our runs
    config={
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_classes": num_classes,
        "criterion": "CrossEntropyLoss",
        "optimizer": "Adam",
    },
)

total_step = len(train_loader)
for epoch in range(num_epochs):
    # train the model on the training set
    train_model(model, train_loader, device, optimizer)

    # evaluate the model on the validation set (this is just for the hyperparameter tuning)
    validation_accuracy = test_model(model, val_loader, device)
    wandb.log({"epoch": epoch, "validation_accuracy": validation_accuracy})

    # ✨ W&B: Create a Table to store predictions for each test step
    columns = ["id", "image", "guess", "truth"]
    for digit in range(10):
        columns.append("score_" + str(digit))
    test_table = wandb.Table(columns=columns)

    # Test the model on the test set
    model.eval()
    log_counter = 0

    test_accuracy = test_model(model, test_loader, device)

    print(
        "Test Accuracy of the model on the 10000 test images: {} %".format(
            test_accuracy
        )
    )
    wandb.log({"epoch": epoch, "test_accuracy": test_accuracy})
    wandb.log({"test_predictions": test_table})

torch.save(model.state_dict(), "model.ckpt")
# We store the model on wandb
artifact = wandb.Artifact(name="model", type="model")
artifact.add_file(local_path="model.ckpt")
wandb.log_artifact(artifact)

wandb.finish()
