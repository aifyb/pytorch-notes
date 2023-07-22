import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from model import VGG19

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(), 
    ])

train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_data_loader = DataLoader(train_data, batch_size=8, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=8, shuffle=True)


model = VGG19().to(device)   # model to GPU
loss_fn = nn.CrossEntropyLoss().to(device)   # loss function to GPU
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 

writer = SummaryWriter('./logs/vgg19')


epochs = 10
for epoch in range(epochs):
    model.train()
    for i, (images, labels) in enumerate(train_data_loader):
        images = images.to(device)   # images to GPU
        labels = labels.to(device)   # labels to GPU
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_data_loader)}], Loss: {loss.item():.4f}')
            writer.add_scalar('loss', loss.item(), epoch * len(train_data_loader) + i)

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_data_loader:
            images = images.to(device)   # images to GPU
            labels = labels.to(device)   # labels to GPU
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy of the model on the test images: {100 * correct / total}%')
        writer.add_scalar('accuracy', 100 * correct / total, epoch)