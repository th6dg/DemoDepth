"""
   THIS FILE CONTAIN FUNCTION WHICH TAKE ALL SEGMENT TENSOR, DISPARITY TENSOR, DEEP TENSOR
"""
import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

# Take segment

def take_all_segment_tensor(folder_path):
  files = os.listdir(folder_path)
  files.sort()
  segment_tensor = []

  for file in files:
    # Define path
    path = folder_path + '/' + file
    # Read segment image
    img = Image.open(path)

    # convert to Tensor
    transform = transforms.Compose([transforms.ToTensor()])

    tensor_from_img = transform(img)
    # Take only 1 channel
    tensor_from_img = tensor_from_img[:1,:,:].float()
    segment_tensor.append(tensor_from_img)
  
  return (segment_tensor)


# Take deep

def take_all_deep_tensor(folder_path):
  files = os.listdir(folder_path)
  files.sort()
  deep_tensor = []

  for index, file in  enumerate(files):
    path = folder_path + '/' + file
    f = open(path, "r")
    contents = f.readlines()
    # Deep element is the first element in line 2
    line_2 = contents[1].strip()
    line_2 = line_2.split()
    deep = float(line_2[0])
    deep_tensor = torch.from_numpy(np.append(deep_tensor, deep))

  return torch.unsqueeze(deep_tensor,1)




# Take disparity with specific index


def take_index_disparity_tensor(index, folder_path):
  
  def generator_disparity():
    files = os.listdir(folder_path)
    files.sort()
    for file in files: 
        df = pd.read_csv(folder_path + '/' + file, header= None)
        yield (df)

  generator = generator_disparity()


  for i, data in enumerate(generator):
    if i != index:
        continue
    else:
        return torch.from_numpy(np.array(data))

        
