import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F #233
import torch.optim as optim
from torchvision import datasets,models,transforms
from PIL import Image
import argparse
import utils
import random

from deeprobust.image.attack.onepixel import Onepixel
import torchvision.transforms as transforms
from deeprobust.image.netmodels import resnet
from deeprobust.image.config import attack_params
from deeprobust.image.utils import download_model
import matplotlib.pyplot as plt
import os.path
import deeprobust.image.netmodels.train_model as trainmodel

random.seed(10)

class AttackedDatasetCIFAR10:

    def __init__(self,attack_rate):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        file_path = './trained_models/CIFAR10_ResNet18_epoch_20.pt'
        if not os.path.exists(file_path):
            trainmodel.train('ResNet18', 'CIFAR10', self.device, 20)
        self.model = Net()

        self.model.load_state_dict(torch.load('./trained_models/' + "CIFAR10_ResNEt18_epoch_20.pt"))
        self.model.eval()
        print("Finish loading network.")

        transform=transforms.Compose([transforms.ToTensor()])

        self.train_data = datasets.CIFAR10('../data', download = True, train= True, transform=transform)
        self.attack_rate = attack_rate
        self.sample_size = round(len(self.train_data.data)*attack_rate/100)

    def generate_adverserial_examples(self, plot=True, plot_path=None ):

        """
        Generate adversarial examples
        """

        indices = torch.randperm(len(self.train_data.data))[:self.sample_size]

        xx = self.train_data.data[indices].to(self.device)
        xx = xx.unsqueeze_(1).float()/255
        #print(xx.size())

        ## Set Target
        yy = self.train_data.targets[indices].to(self.device)

        F1 = Onepixel(self.model, device = self.device)       ### or cpu
        AdvExArray = F1.generate(xx, yy, **attack_params['OnePixel_CIFAR10'])

        predict0 = self.model(xx)
        predict0= predict0.argmax(dim=1, keepdim=True)

        predict1 = self.model(AdvExArray)
        predict1= predict1.argmax(dim=1, keepdim=True)

        print("original prediction:")
        print(predict0)

        print("attack prediction:")
        print(predict1)

        xx = xx.cpu().detach().numpy()
        AdvExArray_np = AdvExArray.cpu().detach().numpy()

        if plot:
            self.plot_adverserial_examples(xx, AdvExArray_np, plot_path)

        return AdvExArray_np, indices

    def plot_adverserial_examples(self, xx, AdvExArray_np, plot_path):
        plt.figure()
        #subplot(r,c) provide the no. of rows and columns
        f, axarr = plt.subplots(2,10, figsize=(20,5))

        # use the created array to output your multiple images. In this case I have stacked 4 images vertically
        for i in range(10):
          axarr[0,i].imshow(xx[i,0]*255,cmap='rgb',vmin=0,vmax=255)
          axarr[1,i].imshow(AdvExArray_np[i,0]*255,cmap='rgb',vmin=0,vmax=255)
        #plt.show()
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);

        utils.checkdir(plot_path)
        f.suptitle('Samples from original CIFAR10 dataset and their attacked versions by OnePixel ')
        axarr[0, 0].set_title('Original Samples')
        axarr[1, 0].set_title('Samples after FGSM attack')
        plt.savefig(plot_path+"attacked_samples.png", dpi=1200, bbox_inches="tight" )
        plt.close()

    def create_adverserial_dataset(self, AdvExArray_np, indices):
        full_data = self.train_data.data
        #print(full_data.size())
        full_data_np = full_data.numpy()
        #print(full_data_np.dtype)

        AdvExArray_np_cp = AdvExArray_np.copy()
        AdvExArray_np_cp = np.reshape(AdvExArray_np_cp,(self.sample_size,32,32))
        #AdvExArray_np_cp *=255
        reshaped3= (AdvExArray_np_cp).astype(np.uint8)

        full_data_np[indices] = reshaped3

        #print(full_data_np[indices])
        #print(full_data_np.dtype)
        np.testing.assert_array_equal(full_data_np[indices],reshaped3)
        return full_data_np

    def full_data_copy(self):
        return self.train_data#.copy()


if __name__ == "__main__":

    attack_rate = 50 # 50% of the train dataset will be attacked
    attack_dataset = AttackedDatasetCIFAR10(attack_rate)

    AdvExArray_np, indices =  attack_dataset.generate_adverserial_examples(plot=True)
    modified_dataset = attack_dataset.create_adverserial_dataset(AdvExArray_np, indices)
    print(len(modified_dataset))
    modified_dataset_pt = attack_dataset.full_data_copy()
    modified_dataset_pt.data = torch.from_numpy(modified_dataset).type(torch.uint8)
    print(modified_dataset_pt.data.size())
    print(modified_dataset_pt.targets.size())
