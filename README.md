# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement

The goal of this project is to develop a Convolutional Neural Network (CNN) for image classification using the Fashion-MNIST dataset. The Fashion-MNIST dataset contains images of various clothing items (T-shirts, trousers, dresses, shoes, etc.), and the model aims to classify them correctly. The challenge is to achieve high accuracy while maintaining efficiency..

## Dataset
![image](https://github.com/user-attachments/assets/279635d2-2e3a-439b-8c95-381363f7e76c)




## Neural Network Model
![Screenshot 2025-03-24 110708](https://github.com/user-attachments/assets/4be359c3-9ad3-4e92-b03d-8a1c66775ad0)



## DESIGN STEPS

### STEP 1: Problem Statement
Define the objective of classifying handwritten digits (0-9) using a Convolutional Neural Network (CNN).

### STEP 2:Dataset Collection
Use the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits.
### STEP 3: Data Preprocessing
Convert images to tensors, normalize pixel values, and create DataLoaders for batch processing.
### STEP 4:Model Architecture
Design a CNN with convolutional layers, activation functions, pooling layers, and fully connected layers.
### STEP 5:Model Training
Train the model using a suitable loss function (CrossEntropyLoss) and optimizer (Adam) for multiple epochs.
### STEP 6:Model Evaluation
Test the model on unseen data, compute accuracy, and analyze results using a confusion matrix and classification report.
### STEP 7: Model Deployment & Visualization
Save the trained model, visualize predictions, and integrate it into an application if needed.
## PROGRAM

### Name:MADESWARAN M
### Register Number:212223040106
```
class CNNClassifier(nn.Module):
  def __init__(self): # Define __init__ method explicitly
    super(CNNClassifier, self).__init__() # Call super().__init__() within __init__
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1) # Correct argument names
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)  # Correct argument names
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1) # Correct argument names
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.fc1 = nn.Linear(128 * 3 * 3, 128) # Adjust input size for Linear layer (Calculation needs update if image size changed)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 10)

  def forward(self, x):
    x = self.pool(torch.relu(self.conv1(x))) # Correctly call self.conv1
    x = self.pool(torch.relu(self.conv2(x)))  # Correctly call self.conv2
    x = self.pool(torch.relu(self.conv3(x))) # Correctly call self.conv3
    x = x.view(x.size(0), -1) # Flatten the tensor
    x = torch.relu(self.fc1(x)) # Correctly call self.fc1
    x = torch.relu(self.fc2(x)) # Correctly call self.fc2
    x = self.fc3(x)
    return x
```

```
# Initialize the Model, Loss Function, and Optimizer
model =CNNClassifier()
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(), lr=0.001)

```

```
# Train the Model
## Step 3: Train the Model
def train_model(model, train_loader, optimizer, criterion, num_epochs=3):
    print('Name: MADESWARAN M')
    print('Register Number: 212223040106')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Print only once per epoch
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')



```

## OUTPUT
### Training Loss per Epoch

![Screenshot 2025-03-24 102057](https://github.com/user-attachments/assets/2837f4b3-b786-4eea-989a-058c4acf5b22)


### Confusion Matrix

![Screenshot 2025-03-24 102157](https://github.com/user-attachments/assets/040c827d-200e-4894-906d-9d924d578c63)


### Classification Report

![Screenshot 2025-03-24 102214](https://github.com/user-attachments/assets/42ccc2b4-0de5-4d9d-a109-5ae56660cb8c)



### New Sample Data Prediction
![Screenshot 2025-03-24 102229](https://github.com/user-attachments/assets/4498207d-f252-4fd0-9413-ede2571d3529)


## RESULT
The CNN model was successfully trained on the MNIST dataset for 3 epochs, achieving a final loss of 0.1632, demonstrating effective handwritten digit classification.
