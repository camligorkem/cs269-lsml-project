import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F #233
import torch.optim as optim
from torchvision import datasets,models,transforms
from PIL import Image
import argparse
import utils

from deeprobust.image.attack.fgsm import FGSM
import torchvision.transforms as transforms
from deeprobust.image.netmodels.CNN import Net
from deeprobust.image.config import attack_params
from deeprobust.image.utils import download_model
import random
import matplotlib.pyplot as plt
import os.path
import deeprobust.image.netmodels.train_model as trainmodel

'''
def parameter_parser():
    parser = argparse.ArgumentParser(description = "Run attack algorithms.")

    parser.add_argument("--destination",
                        default = './trained_models/',
                        help = "choose destination to load the pretrained models.")

    parser.add_argument("--filename",
                        default = "MNIST_CNN_epoch_20.pt")

    return parser.parse_args()

args = parameter_parser() # read argument and creat an argparse object
'''

class AttackedDataset:

    def __init__(self, device='cpu'):
        self.device = device
        file_path = './trained_models/MNIST_CNN_epoch_20.pt'
        if not os.path.exists(file_path):
            trainmodel.train('CNN', 'MNIST', self.device, 20)
        self.model = Net()

        #model.load_state_dict(torch.load(args.destination + args.filename))
        self.model.load_state_dict(torch.load('./trained_models/' + "MNIST_CNN_epoch_20.pt"))
        self.model.eval()
        print("Finish loading network.")

        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

        self.train_data = datasets.MNIST('../data', download = True, train= True, transform=transform)


    def generate_adverserial_examples(self, sample_size, plot=True, plot_path=None ):

        """
        Generate adversarial examples
        """
        indices = torch.randperm(len(self.train_data.data))[:sample_size]

        xx = self.train_data.data[indices].to(self.device)
        xx = xx.unsqueeze_(1).float()#/255  # todo recheck
        #print(xx.size())

        ## Set Target
        yy = self.train_data.targets[indices].to(self.device)

        F1 = FGSM(self.model, device = self.device)       ### or cpu
        AdvExArray = F1.generate(xx, yy, **attack_params['FGSM_MNIST'])

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
            self.plot_adverserial_examples(xx, AdvExArray_np, sample_size, plot_path)

        return AdvExArray_np, indices


    def plot_adverserial_examples(self, xx, AdvExArray_np, sample_size, plot_path):

        '''
        import matplotlib.pyplot as plt
        plt.imshow(xx[0,0]*255,cmap='gray',vmin=0,vmax=255)
        plt.savefig('./adversary_examples/mnist_advexample_fgsm_ori.png')

        plt.imshow(AdvExArray[0,0]*255,cmap='gray',vmin=0,vmax=255)
        plt.savefig('./adversary_examples/mnist_advexample_fgsm_adv.png')
        '''

        plt.figure()

        #subplot(r,c) provide the no. of rows and columns
        f, axarr = plt.subplots(2,10, figsize=(20,5))

        # use the created array to output your multiple images. In this case I have stacked 4 images vertically
        for i in range(10):
          axarr[0,i].imshow(xx[i,0]*255,cmap='gray',vmin=0,vmax=255)
          axarr[1,i].imshow(AdvExArray_np[i,0]*255,cmap='gray',vmin=0,vmax=255)
        #plt.show()
        utils.checkdir(plot_path)
        plt.savefig(plot_path+"attacked_samples.png", dpi=1200)
        plt.close()


    def create_adverserial_dataset(self, AdvExArray_np, indices, sample_size):
        full_data = self.train_data.data
        #print(full_data.size())
        full_data_np = full_data.numpy()
        #print(full_data_np.dtype)
        #full_data_np

        AdvExArray_np_cp = AdvExArray_np.copy()
        AdvExArray_np_cp = np.reshape(AdvExArray_np_cp,(sample_size,28,28))
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
    attack_dataset = AttackedDataset()
    sample_size = 30000
    AdvExArray_np, indices =  attack_dataset.generate_adverserial_examples(sample_size, plot=True)
    modified_dataset = attack_dataset.create_adverserial_dataset(AdvExArray_np, indices, sample_size)
    print(len(modified_dataset))
    modified_dataset_pt = attack_dataset.full_data_copy()
    modified_dataset_pt.data = torch.from_numpy(modified_dataset).type(torch.uint8)
    print(modified_dataset_pt.data.size())
    print(modified_dataset_pt.targets.size())
