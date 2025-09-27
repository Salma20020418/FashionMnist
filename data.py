import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# Custom dataset class
class FashionMNISTDataset(Dataset):
    def __init__(self, x, y, transform=None):
        super().__init__()
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        if self.transform:
            x = self.transform(x)

        return x, y


# Load raw FashionMNIST dataset
train_data = datasets.FashionMNIST(root="./data", train=True, download=True)
test_data = datasets.FashionMNIST(root="./data", train=False, download=True)

# Extract raw data & labels
x_train, y_train = train_data.data, train_data.targets
x_test, y_test = test_data.data, test_data.targets

# Define preprocessing
from torchvision.transforms import ToPILImage

transform = transforms.Compose([
    ToPILImage(),       # convert tensor -> PIL Image
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Wrap datasets
train_dataset = FashionMNISTDataset(x_train, y_train, transform=transform)
test_dataset = FashionMNISTDataset(x_test, y_test, transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

#  Print dataset lengths
print("Length of train dataset:", len(train_dataset))
print("Length of test dataset:", len(test_dataset))

#  Print one batch size
for images, labels in train_loader:
    print("One batch - images shape:", images.shape)
    print("One batch - labels shape:", labels.shape)
    break
