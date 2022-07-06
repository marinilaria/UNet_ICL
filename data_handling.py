import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader,TensorDataset
import torch.optim as optim


class GalaxyDataset(Dataset):
    def __init__(self, image_dir, labels_dir, transform=None):
        self.image_dir = image_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.labels = sorted(os.listdir(labels_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        labels_path = os.path.join(self.labels_dir, self.labels[index])
        images = np.load(img_path)
        for i in range(len(images)):
            image = images[f'arr_{i}.npy']
            image = torch.tensor(image)
        labels = np.load(labels_path)
        for i in range(len(labels)):
            label = labels[f'arr_{i}.npy']
            label = torch.tensor(label)

        image = image.float().unsqueeze(0)
        if self.transform:
            image = self.transform(image)
        image = image.float().squeeze(0)
        return image, label

    def get_loaders(
        data_dir,
        target_maskdir,
        batch_size,
        train_transform,
        num_workers=4,
        pin_memory=False, #this is true if using GPU, but not our case #val_dir, val_maskdir, val_transform,
        randomness = 0,
        size_train = 0.7,
        size_valid = 0.2
        ):

        '''
        This function creates a dataloader.
        How to implement it:
        get_loader(
        train_dir: path of where data in training set are contained,
        train_maskdir: path of where labels in training set are contained,
        batch_size: size of the batch, choice of the user,
        train_transform: tensor containing all the transformations one wants to apply to the data prior the training phase,
        num_workers=4,
        pin_memory=True,  #val_dir, val_maskdir, val_transform,
        randomness = 0, this sets the seed from the random splitting. This way, I can always get the same testing set from
        different runs.
        )'''

        full_dataset = GalaxyDataset(
          image_dir=data_dir,
          labels_dir=target_maskdir,
          transform=train_transform,
        )

        train_size = int(size_train * len(full_dataset))
        valid_size = int(size_valid * len(full_dataset))
        test_size = len(full_dataset) - train_size - valid_size
        torch.manual_seed(randomness)
        train_ds, valid_ds, test_ds = torch.utils.data.random_split(full_dataset, [train_size, valid_size, test_size])
        print ("Proportions of training:validation:testing are :", train_size, valid_size, test_size)

        train_loader = DataLoader(
          train_ds,
          batch_size=batch_size,
          num_workers=num_workers,
          pin_memory=pin_memory,
          shuffle=True,
        )

        valid_loader = DataLoader(
          valid_ds,
          batch_size=batch_size,
          num_workers=num_workers,
          pin_memory=pin_memory,
          shuffle=True,
        )

        test_loader = DataLoader(
          test_ds,
          batch_size=batch_size,
          num_workers=num_workers,
          pin_memory=pin_memory,
          shuffle=True,
        )

        return train_loader, valid_loader, test_loader

    def get_mean_and_std(dataloader):
        channels_sum, channels_squared_sum, num_batches = 0, 0, 0
        for data, _ in dataloader:
            # Mean over batch, height and width, but not over the channels
            channels_sum += data.mean()
            channels_squared_sum += (data**2).mean()
            num_batches += 1

        mean = channels_sum / num_batches

        # std = sqrt(E[X^2] - (E[X])^2)
        std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
        return mean.item(), std.item()


    def train_fn(loader, model, optimizer, loss_fn):
        '''
        This function organizes what happens at each epoch.
        - there is the optimizer set to zero,
        - the model tries to predict the data
        - computation of the loss function wrt the predictions
        - propagation of the information
        - the optimizer takes a step
        '''
        train_loss, points = 0.0, 0
        model.train()
        for batch_idx, (data, targets) in enumerate(loader):
            data = data.float().unsqueeze(1)
            targets = targets.float().unsqueeze(1)

            optimizer.zero_grad()
            predictions = model(data)
            loss = loss_fn(predictions, targets)

            #compute cumulative loss and number of examples used
            train_loss += (loss.item())*data.shape[0]
            points += data.shape[0]

            loss.backward()
            optimizer.step()

        #get the average training loss
        train_loss /= points
        return train_loss


    def valid_fn(loader, model, loss_fn):
        '''
        This function organizes what happens at each epoch.
        - there is the optimizer set to zero,
        - the model tries to predict the data
        - computation of the loss function wrt the predictions
        - propagation of the information
        - the optimizer takes a step
        '''
        valid_loss, points = 0.0, 0
        model.eval()   #set this for validation and testing. Some architecture pieces, e.g. dropout behave differently
        for batch_idx, (data, targets) in enumerate(loader):
            with torch.no_grad():

                data = data.float().unsqueeze(1)
                targets = targets.float().unsqueeze(1)

                predictions = model(data)

                #compute cumulative loss and number of examples used
                valid_loss += (loss_fn(predictions, targets).item())*data.shape[0]
                points += data.shape[0]

        #get the average validation loss
        valid_loss /= points
        return valid_loss

def save_checkpoint(state, filename):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
    
def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])