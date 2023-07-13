'''
    THIS FILE CONTAIN CLASS DATASET
'''
import torch
import sys
sys.path.append('utils')
from get_tensor import take_all_deep_tensor, take_index_disparity_tensor, take_all_segment_tensor


#######################################

class DroneDataset():
  
  def __init__(self, all_segment_tensor, folder_disparity_path, all_deep_tensor ):
    self.all_segment_tensor = all_segment_tensor
    self.folder_disparity_path = folder_disparity_path
    self.all_deep_tensor = all_deep_tensor

  def __getitem__(self, index):
    # Take tensor
    segment = self.all_segment_tensor[index]

    disparity = take_index_disparity_tensor(index, self.folder_disparity_path)

    deep = self.all_deep_tensor[index]

    # Take labels
    if deep < 7:
      label = torch.Tensor([0]).float()
    if 7 <= deep <= 9:
      label = torch.Tensor([1]).float()
    if deep > 9:
      label = torch.Tensor([2]).float()

    return {
        'segment': segment,
        'disparity': disparity,
        'deep' : deep,
        'label' : label
    }

  def __len__(self):
      return self.all_deep_tensor.size(dim = 0)


###############################################

all_deep_tensor = take_all_deep_tensor('/mnt/data/teamAI/duong/2023_6_28_15_19_51-20230711T032031Z-001/2023_6_28_15_19_51/drone_pose')
all_segment_tensor = take_all_segment_tensor('/mnt/data/teamAI/duong/2023_6_28_15_19_51-20230711T032031Z-001/2023_6_28_15_19_51/5/camera_1')

##################################################

dataset = DroneDataset(all_segment_tensor, '/mnt/data/teamAI/duong/disparity', all_deep_tensor)


#################################################

from torch.utils.data import DataLoader
train_loader = DataLoader(dataset , batch_size= 8, shuffle= True)

for index, data in enumerate(train_loader):
  print(data)
  break