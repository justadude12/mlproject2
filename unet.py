import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import f1_score
from PIL import Image
import os
import random

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder
        self.dec4 = self.upconv_block(1024, 512)
        self.dec3 = self.upconv_block(512, 256)
        self.dec2 = self.upconv_block(256, 128)
        self.dec1 = self.upconv_block(128, 64)
        
        # Final Output
        self.final = nn.Conv2d(64, 1, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            self.conv_block(out_channels, out_channels)
        )
    
    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        
        # Decoder path
        dec4 = self.dec4(bottleneck) + enc4
        dec3 = self.dec3(dec4) + enc3
        dec2 = self.dec2(dec3) + enc2
        dec1 = self.dec1(dec2) + enc1
        
        # Final output
        return torch.sigmoid(self.final(dec1))




#=-=-=-=- DATALOADER -=-=-=-=-=-=-=-=-=-=-=-=-=

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        
        # Get sorted list of files to ensure alignment
        self.image_filenames = sorted(os.listdir(images_dir))
        self.mask_filenames = sorted(os.listdir(masks_dir))
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        # Load image and mask
        image_path = os.path.join(self.images_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_filenames[idx])
        
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Grayscale
        
        if self.transform:
            # Apply transformations to both image and mask
            image = self.transform(image)
            mask = self.transform(mask)
        
        # Ensure mask is binary
        mask = (mask > 0.5).float()
        
        return image, mask


class F1Loss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super(F1Loss, self).__init__()
        self.epsilon = epsilon  # Small value to avoid division by zero
    
    def forward(self, preds, targets):
        # Flatten tensors
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        # Compute TP, FP, FN
        tp = (preds * targets).sum()  # True positives
        fp = ((1 - targets) * preds).sum()  # False positives
        fn = (targets * (1 - preds)).sum()  # False negatives
        
        # Compute F1 score components
        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        
        # Return 1 - F1 score to minimize
        return 1 - f1


class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5, epsilon=1e-7):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.bce_loss = nn.BCELoss()
        self.f1_loss = F1Loss(epsilon=epsilon)

    def forward(self, preds, targets):
        bce = self.bce_loss(preds, targets)
        f1 = self.f1_loss(preds, targets)
        return self.alpha * bce + (1 - self.alpha) * f1

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # Flatten tensors
        preds = preds.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        # Calculate Dice coefficient
        intersection = (preds * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        
        return 1 - dice  # Return Dice Loss

class HybridBCEDiceLoss(nn.Module):
    def __init__(self, alpha=0.5, smooth=1.0):
        super(HybridBCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, preds, targets):
        bce = self.bce_loss(preds, targets)
        dice = self.dice_loss(preds, targets)
        return self.alpha * bce + (1 - self.alpha) * dice


##Train
model = UNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = HybridBCEDiceLoss(smooth=0.3) # F1Loss() #nn.BCELoss()
epochs = 1000



transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=70, gamma=0.7)  # Reduce LR every 70 epoch
dataset = SegmentationDataset('/home/ledortz/Road_finding/data/training/images', '/home/ledortz/Road_finding/data/training/groundtruth', transform=transform)
test_size = 16
train_size = len(dataset) - test_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


best_f1_score = 0.0  # Initialize the best F1 score
best_loss = 500.0
model_path = f"unet_model_BCE-DICE_precise_{epochs}.pth"
bestLoss_model_path = f"unet_model_BCE-DICE_precise_flips_{epochs}_bestLoss.pth"
bestF1_model_path = f"unet_model_BCE-DICE_precise_flips_{epochs}_bestF1.pth" 
print(f"-=-Start to train {model_path}-=-") 


for epoch in range(epochs):
    print(f"=-=-=-=-=-=- Epoch {epoch + 1}/{epochs} -=-=-=-=-=-=")

    train_loss = 0.0
    model.train()   
    i = 0
    for images, labels in train_loader:
        
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        del images, labels, outputs
        torch.cuda.empty_cache()

    avg_train_loss = train_loss / len(train_loader)

    print(f"Average Train Loss: {avg_train_loss:.4f}")

    scheduler.step() #Update learning rate every 50 epochs

    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    all_preds_f1 = []
    all_labels_f1 = []
    with torch.no_grad():  # Disable gradient computation for testing
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = (outputs > 0.5).float()  # Binarize predictions

            all_preds_f1.extend(preds.cpu().numpy().flatten())
            all_labels_f1.extend(labels.cpu().numpy().flatten())
	
            # Accumulate test loss
            test_loss += loss.item()

            del images, labels, outputs
            torch.cuda.empty_cache()

        f1 = f1_score(all_labels_f1, all_preds_f1)
        print(f"F1 Score: {f1:.4f}")

        if f1 > best_f1_score:
            best_f1_score = f1
            torch.save(model.state_dict(), bestF1_model_path)

    # Average test loss for this epoch
    avg_test_loss = test_loss / len(train_loader)
    if avg_test_loss < best_loss:
        best_loss = avg_test_loss
        torch.save(model.state_dict(), bestLoss_model_path)


    print(f"Average Testing Loss: {avg_test_loss:.4f}") 


torch.save(model.state_dict(), model_path)

print('best F1 score is : ' + str(best_f1_score)) 
print('best loss is : ' + str(best_loss)) 