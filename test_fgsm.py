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
from tqdm import tqdm

from deeprobust.image.attack.fgsm import FGSM
import torchvision.transforms as transforms
from deeprobust.image.netmodels.CNN import Net
from deeprobust.image.config import attack_params
from deeprobust.image.utils import download_model
import matplotlib.pyplot as plt
import os.path
import deeprobust.image.netmodels.train_model as trainmodel

random.seed(10)

class AttackedDataset:

    def __init__(self, attack_rate):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        file_path = './trained_models/MNIST_CNN_epoch_20.pt'
        if not os.path.exists(file_path):
            trainmodel.train('CNN', 'MNIST', self.device, 20)
        self.model = Net()

        #model.load_state_dict(torch.load(args.destination + args.filename))
        self.model.load_state_dict(torch.load('./trained_models/' + "MNIST_CNN_epoch_20.pt"))
        self.model.eval()
        print("Finish loading network.")

        transform=transforms.Compose([transforms.ToTensor()])

        self.train_data = datasets.MNIST('../data', download = True, train= True, transform=transform)
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
            self.plot_adverserial_examples(xx, AdvExArray_np, plot_path)

        return AdvExArray_np, indices

    def generate_full_adverserial_dataset(self, plot=False, plot_path="" ):

        """
        Generate full adversarial dataset
        """

        xx = self.train_data.data.to(self.device)
        xx = xx.unsqueeze_(1).float()/255
        # print(xx.size())

        ## Set Target
        yy = self.train_data.targets.to(self.device)

        F1 = FGSM(self.model, device = self.device)       ### or cpu

        AdvExArray = xx

        batch_size = 60
        iter_num = int(len(self.train_data.data)/batch_size)
        for b in tqdm(range(0, iter_num)):
            #print(batch_size*b,batch_size*b+(batch_size))
            small_xx = xx[batch_size*b:batch_size*b+(batch_size)]
            small_yy = yy[batch_size*b:batch_size*b+(batch_size)]
            AdvExArray_small = F1.generate(small_xx, small_yy, **attack_params['FGSM_MNIST'])
            AdvExArray[batch_size*b:batch_size*b+(batch_size)] = AdvExArray_small

        # torch.cuda release cache
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        xx = xx.cpu().detach().numpy()
        AdvExArray_np = AdvExArray.cpu().detach().numpy()
        utils.checkdir("../data/MNIST_fsm_attack/")

        # save attacked dataset
        np.savez_compressed('../data/MNIST_fsm_attack/attacked_train_data.npz', data=AdvExArray_np)

        loaded = np.load('../data/MNIST_fsm_attack/attacked_train_data.npz')
        assert(np.array_equal(AdvExArray_np, loaded['data']))

        if plot:
            indices = torch.randperm(len(self.train_data.data))[:10]
            self.plot_adverserial_examples(xx[indices], AdvExArray_np[indices], plot_path)


    def plot_adverserial_examples(self, xx, AdvExArray_np, plot_path):
        plt.figure()
        #subplot(r,c) provide the no. of rows and columns
        f, axarr = plt.subplots(2,10, figsize=(20,5))

        # use the created array to output your multiple images. In this case I have stacked 4 images vertically
        for i in range(10):
          axarr[0,i].imshow(xx[i,0]*255,cmap='gray',vmin=0,vmax=255)
          axarr[1,i].imshow(AdvExArray_np[i,0]*255,cmap='gray',vmin=0,vmax=255)
        #plt.show()
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);

        utils.checkdir(plot_path)
        f.suptitle('Samples from original MNIST dataset and their attacked versions by FGSM ')
        axarr[0, 0].set_title('Original Samples')
        axarr[1, 0].set_title('Samples after FGSM attack')
        plt.savefig(plot_path+"attacked_samples.png", dpi=1200, bbox_inches="tight" )
        plt.close()

    def create_partial_adverserial_dataset(self, attack_rate, plot, plot_path):
        attacked_dataset_path = '../data/MNIST_fsm_attack/attacked_train_data.npz'
        if not os.path.exists(attacked_dataset_path):
            # create attacked dataset
            self.generate_full_adverserial_dataset(plot=plot, plot_path = plot_path)

        # read from created attacked dataset
        loaded = np.load(attacked_dataset_path)
        full_attacked_dataset = loaded['data']

        # get samples from attacked datasets
        sample_size = round(len(full_attacked_dataset)*attack_rate/100)
        attacked_indices = torch.randperm(len(full_attacked_dataset))[:sample_size]

        attacked_samples = full_attacked_dataset[attacked_indices]
        attacked_samples = np.reshape(attacked_samples,(sample_size,28,28))
        attacked_samples_torch =  torch.from_numpy(attacked_samples).type(torch.uint8)

        full_original_data = self.train_data
        # print(full_original_data.data.size())
        full_original_data.data[attacked_indices] = attacked_samples_torch
        torch.equal(full_original_data.data[attacked_indices], attacked_samples_torch)

        return full_original_data


if __name__ == "__main__":
    attack_rate = 50 # 50% of the train dataset will be attacked
    attack_dataset = AttackedDataset(attack_rate)

    data = attack_dataset.create_partial_adverserial_dataset(attack_rate, plot=False)
    print(len(data))
    print(data.data.size())
    print(data.targets.size())
    print('Done')
