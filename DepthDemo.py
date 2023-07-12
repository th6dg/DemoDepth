import os
def take_path_from_folder(index,folder_path):

    # Open a folder
    path = os.listdir( folder_path )
    return path[index]

################################################

from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

def take_segment_tensor(path):
  # Read segment image
  img = Image.open(path)

  # convert to Tensor
  transform = transforms.Compose([transforms.ToTensor()])

  tensor_from_img = transform(img)
  # Take only 1 channel for reducing cost computing
  tensor_from_img = tensor_from_img[:1,:,:].float()
  return (tensor_from_img)


# Take disparity

def take_disparity_tensor(path):
  # Read file txt
  with open(path) as f:
      contents = f.readlines()
  f.close()

  # Process string data and convert to 2D array
  arr = []

  for line in contents:
    line = line.strip()
    line = line.replace("[","").replace("]","")
    x = line.split()
    x = np.array(x,dtype = 'float')
    arr = np.append(arr,x)

  arr = arr.reshape(480, 720)
  arr = torch.from_numpy(arr)
  arr = torch.unsqueeze(arr,0).float()
  return (arr)



# Take deep

def take_deep_tensor(path):
  # Read file txt
  with open(path) as f:
      contents = f.readlines()
  f.close()

  # Process string data and convert to 2D array
  arr = []

  for line in contents:
    line = line.strip()
    x = line.split()
    x = np.array(x)
    arr = np.append(arr,x)
  out = np.array(arr[7], dtype= 'float')
  out = torch.from_numpy(out)
  out = torch.unsqueeze(out,0).float()
  return out


###################################################
class DroneDataset():
  def __init__(self, segment_folder_path, disparity_folder_path, deep_folder_path):
    self.segment_folder_path = segment_folder_path
    self.disparity_folder_path = disparity_folder_path
    self.deep_folder_path = deep_folder_path

  def __getitem__(self, index):
    # Take path
    segment_path =  self.segment_folder_path + '/' + take_path_from_folder(index, self.segment_folder_path)
    disparity_path =  self.disparity_folder_path + '/' + take_path_from_folder(index, self.disparity_folder_path)
    deep_path = self.deep_folder_path + '/' + take_path_from_folder(index, self.deep_folder_path)

    # Take tensor from image
    segment = take_segment_tensor(segment_path)
    disparity = take_disparity_tensor(disparity_path)
    deep = take_deep_tensor(deep_path)

    # Take labels
    if deep < 7:
      label = torch.Tensor([0]).long()
    if 7 <= deep <= 9:
      label = torch.Tensor([1]).long()
    if deep > 9:
      label = torch.Tensor([2]).long()

    return {
        'segment': segment,
        'disparity': disparity,
        'deep' : deep,
        'label' : label
    }

  def __len__(self):
      return len(os.listdir(self.segment_folder_path))
  


########################################################
from torch.utils.data import DataLoader

dataset = DroneDataset('/mnt/data/teamAI/duong/2023_6_28_15_19_51-20230711T032031Z-001/2023_6_28_15_19_51/5/camera_1',
                       '/mnt/data/teamAI/duong/disparity',
                       '/mnt/data/teamAI/duong/2023_6_28_15_19_51-20230711T032031Z-001/2023_6_28_15_19_51/drone_pose')
# Dataloader
train_loader = DataLoader(dataset, batch_size = 32, shuffle = True)


#####################################################
import torch.nn as nn
from torchvision.models import resnet18

backbone1 = resnet18()
backbone1.conv1 = nn.Conv2d(1,64, kernel_size=(7,7),stride = 2)
#backbone2 = resnet18()

class Head(nn.Module):
  def __init__(self):
    super(Head, self).__init__()
    self.x1 = nn.Sequential(
      nn.Linear(in_features= 2000, out_features= 256),
      nn.ReLU(),
      nn.Linear(in_features= 256, out_features= 3),
      nn.Softmax(dim = 1)
      )
    self.dense1 = nn.Sequential(
        nn.Linear(in_features=2000, out_features=256),
        nn.ReLU(),
    )
    self.dense2 = nn.Linear(in_features=259, out_features=1)

  def forward(self, t):
    out_1 = self.x1(t)
    out_2 = self.dense1(t)
    out_2 = torch.cat((out_1,out_2), dim=1)
    out = self.dense2(out_2)

    return out_1, out


class TotalModel(nn.Module):
  def __init__(self):
    super(TotalModel, self).__init__()
    self.backbone1 = backbone1
    self.head = Head()

  def forward(self,x,y):
    x = self.backbone1(x)
    y = self.backbone1(y)
    x = torch.cat((x,y),dim = 1)
    out_1, out = self.head(x)

    return out_1, out
  
#################################################
model = TotalModel()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.train()


####################################################
# Loss, Optimizer
depth_loss_fn = nn.MSELoss(reduction = 'sum')
label_loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr= 0.001)

##############################################
# Training

epochs = 10
iter = 0
fig = plt.figure()
for epoch in range(int(epochs)):
    loss_ = 0
    for index, sample in enumerate(train_loader):
        segment = sample['segment'].cuda()
        disparity = sample['disparity'].cuda()
        deep = sample['deep'].cuda()
        label = sample['label'].cuda()
        label = torch.squeeze(label,dim =1)


        optimizer.zero_grad()

        #FEED DATA FOR MODEL
        label_pre, depth = model(disparity, segment)
        #print(f' l: {label_pre.shape}, d: {depth.shape}, i:{label.shape}, j:{deep.shape}')

        # Compute the predicted labels
        _, l_pre = torch.max(label_pre.data, 1)

        depth_loss = depth_loss_fn(depth, deep)
        label_loss = label_loss_fn(label_pre, label)
        loss = 0.000 * label_loss + depth_loss
        loss_ += loss.item()
        loss.backward()
        optimizer.step()
        iter+=1

        if index % 5 ==0 :
          print(f'example_pre : {depth}, ground_truth: {deep}')
          print(f'label_pre: {l_pre}, label_gt: {label}')
        print(f'Done index: {index}!')


    print(f'Loss: {loss_/1000}')