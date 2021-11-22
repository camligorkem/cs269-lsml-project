import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,models,transforms
from PIL import Image

from deeprobust.image.attack.fgsm import FGSM
from deeprobust.image.attack.pgd import PGD
import deeprobust.image.netmodels.resnet as resnet
import deeprobust.image.netmodels.CNN as CNN
from deeprobust.image.config import attack_params
import matplotlib.pyplot as plt
import os.path
import deeprobust.image.netmodels.train_model as trainmodel
from deeprobust.image.utils import download_model
import random
import utils
from tqdm import tqdm

random.seed(10)

class CIFAR10_AttackDataset:


    def __init__(self, attack_rate, attack_type):
        print(f'CIFAR10, {attack_type}')
        self.attack_type = attack_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        file_path = './trained_models/CIFAR10_ResNet18_epoch_20.pt'
        if not os.path.exists(file_path):
            URL = "https://github.com/I-am-Bot/deeprobust_model/raw/master/CIFAR10_ResNet18_epoch_20.pt"
            download_model(URL, file_path)
            #trainmodel.train('CIFAR10', 'ResNet18', self.device, 20)
        self.model = resnet.ResNet18().to('cpu')
        print("Load network")

        #model.load_state_dict(torch.load(args.destination + args.filename))
        self.model.load_state_dict(torch.load("./trained_models/CIFAR10_ResNet18_epoch_20.pt",map_location=torch.device('cpu')))
        self.model.eval()
        self.get_configs(attack_type)

        print("Finish loading network.")

        transform_cifar10= transforms.Compose([
                    #transforms.RandomCrop(32, padding=5),
                    #transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])
        self.batch_size = 60
        self.train_data = datasets.CIFAR10(root='../data', train=True,
                                        download=True, transform=transform_cifar10)

        self.train_loader = torch.utils.data.DataLoader(self.train_data,
                                             batch_size = self.batch_size, shuffle=False) #, **kwargs)
        #print(len(self.train_loader))
        #print(len(self.train_loader.dataset))

        self.attack_rate = attack_rate
        self.sample_size = round(len(self.train_loader.dataset)*attack_rate/100)
        self.classes = np.array(('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'))

    def get_configs(self, attack_type):
        if attack_type =='fgsm':
            self.adversary_model = FGSM(self.model, device = self.device)

        elif attack_type == 'pgd':
            self.adversary_model = PGD(self.model, device = self.device)
        else:
            print("\nWrong Dataset choice \n")
            exit()

    def generate_full_adverserial_dataset(self, plot=False, plot_path="./" ):

        """
        Generate full adversarial dataset
        """
        # to do check here!
        #xx, yy = next(iter(self.train_loader))
        sample_size = 50000
        xx = torch.Tensor(self.train_data.data)#.to(self.device).float()
        xx = torch.reshape(xx,(sample_size,3,32,32))
        xx = xx.to('cpu')#.float()

        yy = torch.Tensor(self.train_data.targets)

        AdvExArray = xx
        iter_num = int(len(xx)/self.batch_size)
        for b in tqdm(range(0, iter_num)):
            #print(batch_size*b,batch_size*b+(batch_size))
            small_xx = xx[self.batch_size*b:self.batch_size*b+(self.batch_size)]
            small_yy = yy[self.batch_size*b:self.batch_size*b+(self.batch_size)]
            if self.attack_type == 'pgd':
                AdvExArray_small = self.adversary_model.generate(small_xx, small_yy, **attack_params['PGD_CIFAR10'])#.float()
            else:
                AdvExArray_small = self.adversary_model.generate(small_xx, small_yy)
            AdvExArray[self.batch_size*b:self.batch_size*b+(self.batch_size)] = AdvExArray_small


        # torch.cuda release cache
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        xx = xx.cpu().detach().numpy()
        AdvExArray_np = AdvExArray.cpu().detach().numpy()

        print('full adverserial data len:', len(AdvExArray_np))

        utils.checkdir(f"../data/CIFAR10_{self.attack_type}_attack/")

        # save attacked dataset
        np.savez_compressed(f'../data/CIFAR10_{self.attack_type}_attack/attacked_train_data.npz', data=AdvExArray_np)

        loaded = np.load(f'../data/CIFAR10_{self.attack_type}_attack/attacked_train_data.npz')
        assert(np.array_equal(AdvExArray_np, loaded['data']))

        if plot:
            indices = torch.randperm(len(self.train_loader.dataset))[:10]
            self.plot_adverserial_examples(xx[indices], AdvExArray_np[indices], plot_path)
            #self.plot_adverserial_examples(xx[:10], AdvExArray_np[:10], plot_path)

    def plot_adverserial_examples(self, xx, AdvExArray, plot_path=''):
        plt.figure()
        #subplot(r,c) provide the no. of rows and columns
        f, axarr = plt.subplots(2,10, figsize=(20,5))

        # use the created array to output your multiple images. In this case I have stacked 4 images vertically
        for i in range(10):
            x_show_sample =  np.reshape(xx,(10,32,32,3))[i]
            axarr[0,i].imshow(x_show_sample/255,vmin=0,vmax=255)

            AdvExArray_sample = np.reshape(AdvExArray,(10,32,32,3))[i]
            axarr[1,i].imshow(AdvExArray_sample/255,vmin=0,vmax=255)

        #plt.show()
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);

        utils.checkdir(plot_path)
        f.suptitle(f'Samples from original CIFAR10 dataset and their attacked versions by {self.attack_type} ')
        axarr[0, 0].set_title('Original Samples')
        axarr[1, 0].set_title(f'Samples after {self.attack_typeattack_type} attack')
        plt.savefig(plot_path+"attacked_samples.png", dpi=1200, bbox_inches="tight" )
        plt.close()
        print('saved plottt')


    def create_partial_adverserial_dataset(self, attack_rate, plot, plot_path='./', recreate=False):
        attacked_dataset_path = f'../data/CIFAR10_{self.attack_type}_attack/attacked_train_data.npz'
        if (not os.path.exists(attacked_dataset_path)) or recreate:
            # create attacked dataset
            self.generate_full_adverserial_dataset(plot=plot, plot_path = plot_path)

        # read from created attacked dataset
        loaded = np.load(attacked_dataset_path)
        full_attacked_dataset = loaded['data']

        print(len(full_attacked_dataset))
        #print(full_attacked_dataset.size)
        # get samples from attacked datasets
        sample_size = round(len(full_attacked_dataset)*attack_rate/100)
        attacked_indices = torch.randperm(len(full_attacked_dataset))[:sample_size]

        attacked_samples = full_attacked_dataset[attacked_indices]
        # print(attacked_samples.size())
        attacked_samples = np.reshape(attacked_samples,(sample_size,32,32,3))
        attacked_samples = np.uint8(attacked_samples)
        #attacked_samples_torch =  torch.from_numpy(attacked_samples).type(torch.uint8)

        full_original_data = self.train_data
        #print(type(full_original_data))
        print(type(full_original_data.data))
        full_original_data.data[attacked_indices] = attacked_samples
        np.testing.assert_array_equal(full_original_data.data[attacked_indices], attacked_samples)

        return full_original_data


if __name__ == "__main__":
    attack_rate = 50 # 50% of the train dataset will be attacked
    attack_type = 'fgsm'#'pgd'
    pgd_attack_dataset = CIFAR10_AttackDataset(attack_rate,attack_type)

    data = pgd_attack_dataset.create_partial_adverserial_dataset(attack_rate, plot=True, recreate=True)
    print(len(data))
    print('Done')
