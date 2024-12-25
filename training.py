import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import json
import timm
import torch.nn as nn

class MojaMreza(torch.nn.Module):
    def __init__(self, num_classes):
        super(MojaMreza, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        enet_out_size = 1280
        # Make a classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enet_out_size, num_classes)
        )

    def forward(self, x):
        # Connect these parts and return the output
        x = self.features(x)
        output = self.classifier(x)
        return output

class MojaMrezaOG(torch.nn.Module):
    def __init__(self, num_classes):
        super(MojaMrezaOG, self).__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2, 2),

            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2, 2),

            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2, 2)
        )
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 8 * 8, 128),  # Adjust input size to 64 * 8 * 8
            torch.nn.Tanh(),
            torch.nn.Linear(128, num_classes),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x): #
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.data = ImageFolder(img_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __get_categories__(self):
        return self.data.classes

def write_labels(data_path):
    target_to_class = {v: k for k, v in ImageFolder(data_path).class_to_idx.items()}
    print(target_to_class)
    json_object = json.dumps(target_to_class, indent=4)
    with open("labels.json", "w") as outfile:
        outfile.write(json_object)

if __name__ == '__main__':
    # transform, ki bomo obdelali nad slikami, v našem primeru so že prave velikosti
    transform_train = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # odperemo oba dataseta, ime folderja predstavlja label
    train_data_path = "train_data/"
    val_data_path = "val_data/"

    train_dataset = CustomImageDataset(train_data_path, transform_train)
    val_dataset = CustomImageDataset(val_data_path, transform_test)

    # izpišemo lable
    print(train_dataset.__get_categories__())
    write_labels(train_data_path)

    # Parametri
    num_classes = len(train_dataset.__get_categories__()) # št različnih stvari, ki jih bomo razpoznovali = št leblov
    batch_size = 32 # koliko slik naenkrat bomo vzeli nenkrat za treniranje
    epochs = 20 # koliko zagonov učenja bomo meli
    learning_rate = 0.001

    # ustvarimo classe za nalaganje slik
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # ustvarimo instanco modela
    model = MojaMreza(num_classes)

    # inicalizacija
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # najde najboljši komplet parametrov z pomočjo crietion-a

    # TensorBoard setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    train_losses, val_losses = [], []

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_data_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss /= train_data_loader.__len__()
        train_losses.append(train_loss)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_data_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

        val_loss /= train_data_loader.__len__()
        val_losses.append(val_loss)

        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    # shranimo model, za kasnejšo testiranje
    torch.save(model.state_dict(), "model_0_timm.pth")

    # pokažemo graf izgub, skozi vsaki run (epoch)
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()
    plt.title("Loss over epochs")
    plt.show()