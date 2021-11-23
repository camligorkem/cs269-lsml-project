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
from deeprobust.image.attack.pgd import PGD
import torchvision.transforms as transforms
from deeprobust.image.netmodels.CNN import Net
from deeprobust.image.config import attack_params
from deeprobust.image.utils import download_model
import matplotlib.pyplot as plt
import os.path
import deeprobust.image.netmodels.train_model as trainmodel

random.seed(10)

class MNIST_AttackedDataset:

    def __init__(self, attack_rate, attack_type):
        self.attack_type= attack_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        utils.checkdir('./trained_models/')
        file_path = './trained_models/MNIST_CNN_epoch_20.pt'
        if not os.path.exists(file_path):
            URL = "https://github.com/I-am-Bot/deeprobust_model/raw/master/MNIST_CNN_epoch_20.pt"
            download_model(URL, file_path)
            #trainmodel.train('CNN', 'MNIST', self.device, 20)
        self.model = Net()

        #model.load_state_dict(torch.load(args.destination + args.filename))
        self.model.load_state_dict(torch.load('./trained_models/' + "MNIST_CNN_epoch_20.pt"))
        self.model.eval()
        self.get_configs(attack_type)
        print("Finish loading network.")

        transform=transforms.Compose([transforms.ToTensor()])

        self.train_data = datasets.MNIST('../data', download = True, train= True, transform=transform)
        self.attack_rate = attack_rate
        self.sample_size = round(len(self.train_data.data)*attack_rate/100)

    def get_configs(self, attack_type):
        if attack_type =='fgsm':
            self.adversary_model = FGSM(self.model, device = self.device)

        elif attack_type == 'pgd':
            self.adversary_model = PGD(self.model, device = self.device)
        else:
            print("\nWrong attack choice \n")
            exit()

    def generate_full_adverserial_dataset(self, plot=False, plot_path="" ):

        """
        Generate full adversarial dataset
        """

        xx = self.train_data.data.to(self.device)
        xx = xx.unsqueeze_(1).float()/255
        # print(xx.size())

        ## Set Target
        yy = self.train_data.targets.to(self.device)

        AdvExArray = torch.clone(xx)

        batch_size = 60
        iter_num = int(len(self.train_data.data)/batch_size)
        for b in tqdm(range(0, iter_num)):
            #print(batch_size*b,batch_size*b+(batch_size))
            small_xx = xx[batch_size*b:batch_size*b+(batch_size)]
            small_yy = yy[batch_size*b:batch_size*b+(batch_size)]
            if self.attack_type == 'fgsm':
                AdvExArray_small = self.adversary_model.generate(small_xx, small_yy, **attack_params['FGSM_MNIST'])#.float()
            elif self.attack_type == 'pgd':
                params = {
                    'epsilon': 0.1,
                    'clip_max': 1.0,
                    'clip_min': 0.0,
                    'print_process': False
                    }
                AdvExArray_small = self.adversary_model.generate(small_xx, small_yy, **params)
            else:
                print("\nWrong attack choice \n")
                exit()

            AdvExArray[batch_size*b:batch_size*b+(batch_size)] = AdvExArray_small

        # torch.cuda release cache
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        xx = xx.cpu().detach().numpy()
        AdvExArray_np = AdvExArray.cpu().detach().numpy()
        utils.checkdir(f"../data/MNIST_{self.attack_type}_attack/")

        # save attacked dataset
        np.savez_compressed(f'../data/MNIST_{self.attack_type}_attack/attacked_train_data.npz', data=AdvExArray_np)

        loaded = np.load(f'../data/MNIST_{self.attack_type}_attack/attacked_train_data.npz')
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
        f.suptitle(f'Samples from original MNIST dataset and their attacked versions by {self.attack_type} ')
        axarr[0, 0].set_title('Original Samples')
        axarr[1, 0].set_title(f'Samples after {self.attack_type} attack')
        plt.savefig(plot_path+"attacked_samples.png", dpi=1200, bbox_inches="tight" )
        plt.close()

    def create_partial_adverserial_dataset(self, attack_rate, plot, plot_path, recreate=False):
        attacked_dataset_path = f'../data/MNIST_{self.attack_type}_attack/attacked_train_data.npz'
        if (not os.path.exists(attacked_dataset_path)) or recreate:
            # create attacked dataset
            self.generate_full_adverserial_dataset(plot=plot, plot_path = plot_path)

        # read from created attacked dataset
        loaded = np.load(attacked_dataset_path)
        full_attacked_dataset = loaded['data']

        # get samples from attacked datasets
        sample_size = round(len(full_attacked_dataset)*attack_rate/100)
        attacked_indices = torch.randperm(len(full_attacked_dataset))[:sample_size]

        attacked_samples = full_attacked_dataset[attacked_indices]
        # print(attacked_samples.size())
        attacked_samples = np.reshape(attacked_samples,(sample_size,28,28))
        attacked_samples_torch =  torch.from_numpy(attacked_samples).type(torch.uint8)

        full_original_data = self.train_data
        #print(full_original_data.data.size())
        #print(attacked_samples_torch.size())
        full_original_data.data[attacked_indices] = attacked_samples_torch
        torch.equal(full_original_data.data[attacked_indices], attacked_samples_torch)

        return full_original_data


if __name__ == "__main__":
    attack_rate = 50 # 50% of the train dataset will be attacked
    attack_type = 'fgsm'
    attack_dataset = MNIST_AttackedDataset(attack_rate, attack_type)

    data = attack_dataset.create_partial_adverserial_dataset(attack_rate, plot=False)
    print(len(data))
    print(data.data.size())
    print(data.targets.size())
    print('Done')
