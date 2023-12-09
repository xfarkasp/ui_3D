import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

def metric_calc(model_name,true_labels, predicted_labels, conf_matrix):
    # conf metrix visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    accuracy = accuracy_score(true_labels, predicted_labels) * 100
    precision = precision_score(true_labels, predicted_labels, average='weighted') * 100
    recall = recall_score(true_labels, predicted_labels, average='weighted') * 100
    f1 = f1_score(true_labels, predicted_labels, average='weighted') * 100

    metrics = ['Accuracy' ,'Precision', 'Recall', 'F1-Score']
    values = [accuracy, precision, recall, f1]

    plt.figure(figsize=(12, 6))

    plt.bar(metrics, values, color=['blue', 'red', 'green', 'orange'])
    plt.ylim(0, 100)  # Assuming values are between 0 and 1

    for i, value in enumerate(values):
        plt.text(i, value + 0.05, f'{value:.4f}', ha='center', va='center', color='black')

    plt.xlabel("Metrics")
    plt.ylabel("Values")
    plt.title(model_name)
    plt.show()

# Feedforward
class Feed_Forward(nn.Module):

    def __init__(self):
        super(Feed_Forward, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        self.model_name = "FeedForward"

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Cnn
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 14 * 14, 128)  # Adjusted input size
        self.fc2 = nn.Linear(128, 10)
        self.model_name = "CNN"

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Deep neural network
class Deep(nn.Module):
    def __init__(self):
        super(Deep, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, 10)
        self.model_name = "Deep"

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
    print(f"Accuracy score is: {accuracy}%")
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    metric_calc(model.model_name, all_labels, all_predictions, conf_matrix)


def classify(model):
    path = input("Path to image or 0 for return: ")
    if path == '0':
        return

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
    plt.title(f'Predicted Number: {predicted_class.item()}')
    plt.show()


if __name__ == "__main__":
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
    # X_train_tensor = torch.from_numpy(X_train.values).float()
    X_train_tensor = torch.from_numpy(X_train.values.reshape(-1, 1, 28, 28)).float()
    y_train_tensor = torch.from_numpy(y_train.values).long()

    # X_test_tensor = torch.from_numpy(X_test.values).float()
    X_test_tensor = torch.from_numpy(X_test.values.reshape(-1, 1, 28, 28)).float()
    y_test_tensor = torch.from_numpy(y_test.values).long()

    # Create DataLoader for training and testing sets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    current_model = None

    while True:
        print(
            "Menu:\n 1.Train model\n 2.Load model\n 3.Test model\n 4. Train and Test model\n 5.Classify snipets\n 6.Save model")
        user_input = input("Input: ")
        if user_input == '1':
            print("Models:\n 1. Feed Forward \n 2. CNN \n 3. Deep")
            model_select = input("Chosen model: ")
            if model_select == '1':
                model = Feed_Forward()

                # loss function and optimizer
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)

                train_model(model, optimizer, train_loader)
                current_model = model
            elif model_select == '2':
                model = CNN()

                # loss function and optimizer
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)

                train_model(model, optimizer, train_loader)
                current_model = model

            elif model_select == '3':

                X_test_tensor = torch.from_numpy(X_test.values.reshape(-1, 1, 28, 28)).float()
                y_test_tensor = torch.from_numpy(y_test.values).long()

                model = Deep()

                # loss function and optimizer
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)

                train_model(model, optimizer, train_loader)
                current_model = model

        # elif user_input == '2':
        #     load_path = input("Path to model: ")
        #     current_model = torch.load(load_path)

        elif user_input == '3':
            if current_model != None:
                evaluate_model(current_model, test_loader)
            else:
                print("No model was trained or loaded")

        elif user_input == '4':
            print("Models:\n 1. Feed Forward \n 2. CNN \n 3. Deep")
            model_select = input("Chosen model: ")
            if model_select == '1':
                model = Feed_Forward()

                # loss function and optimizer
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)

                train_model(model, optimizer, train_loader)
                current_model = model
                evaluate_model(current_model, test_loader)

        elif user_input == '5':
            while True:
                classify(current_model)

        elif user_input == '6':
            if current_model != None:
                save_path = input("Path to save model")
                torch.save(current_model.state_dict(), save_path + current_model.model_name +'.pth')
            else:
                print("No model was trained or loaded")

        elif user_input == '0':
            break