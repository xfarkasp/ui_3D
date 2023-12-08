import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

save_path = "C:\\Users\\pedro\\PycharmProjects\\ui_3D\\models\\"

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Load the MNIST dataset
mnist = fetch_openml('mnist_784')
X, y = mnist.data.astype('float32'), mnist.target.astype('int64')

# Normalize pixel values to be between 0 and 1
X /= 255.0

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert NumPy arrays to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train.values).float()
y_train_tensor = torch.from_numpy(y_train.values).long()

X_test_tensor = torch.from_numpy(X_test.values).float()
y_test_tensor = torch.from_numpy(y_test.values).long()

# Create DataLoader for training and testing sets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Model 1: Simple Feedforward Neural Network
class Feed_Forward(nn.Module):

    def __init__(self):
        super(Feed_Forward, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

feed_forward = Feed_Forward()

# Model 2: Convolutional Neural Network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(14 * 14 * 32, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model2 = CNN()

# Model 3: Deep Feedforward Neural Network
class DeepNN(nn.Module):
    def __init__(self):
        super(DeepNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x

model3 = DeepNN()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(feed_forward.parameters(), lr=0.001)


# Train the models
def train_model(model, optimizer, train_loader, epochs=5):
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')
    torch.save(model.state_dict(), 'model_state_dict.pth')


# Training each model
train_model(feed_forward, optimizer1, train_loader)

# Evaluate the models on the test set
def evaluate_model(model, test_loader):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            all_predictions.extend(predictions.numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    return accuracy, all_labels, all_predictions

# Assuming 'feed_forward' is your trained model and 'test_loader' is your DataLoader
feed_forward_accuracy, true_labels, predicted_labels = evaluate_model(feed_forward, test_loader)

print(f'Feed Forward Accuracy: {feed_forward_accuracy * 100:.2f}%')


def class_accuracy(true_labels, predicted_labels, num_classes):
    class_acc = []
    for i in range(num_classes):
        class_true = np.array(true_labels) == i
        class_pred = np.array(predicted_labels) == i
        acc = accuracy_score(class_true, class_pred)
        class_acc.append(acc)
    return class_acc

class_accuracies = class_accuracy(true_labels, predicted_labels, num_classes=10)

# Bar Chart for Overall Accuracy
plt.figure(figsize=(8, 6))
sns.barplot(x=['Feed Forward'], y=[feed_forward_accuracy], color='skyblue')
plt.ylim(0, 1)  # Set y-axis limit to represent accuracy percentage
plt.ylabel("Accuracy")
plt.title("Overall Accuracy")
plt.show()

def clasify(model):
    path = input("Path to image: ")
    image = Image.open(path).convert('L')  # Convert to grayscale
    transform = transforms.Compose([transforms.ToTensor()])
    input_image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(input_image)

    # Get the predicted class
    _, predicted_class = torch.max(output, 1)

    print(f'The predicted class is: {predicted_class.item()}')
    # Show the image
    plt.imshow(image, cmap='gray')
    plt.title(f'Predicted Class: {predicted_class.item()}')
    plt.show()

while True:
    clasify(feed_forward)