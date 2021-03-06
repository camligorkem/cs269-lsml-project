# Importing Libraries
import argparse
import copy
import os
import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import os
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import seaborn as sns
import torch.nn.init as init
import pickle
#from torch.utils.data.sampler import SubsetRandomSampler
from mnist_attack_fgsm_pgd import MNIST_AttackedDataset
from cifar10_attack_fgsm_pgd import CIFAR10_AttackDataset

# Custom Libraries
import utils
import random

# Tensorboard initialization
writer = SummaryWriter()

# Plotting Style
sns.set_style('darkgrid')

random.seed(10)

# Main
def main(args, ITE=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reinit = True if args.prune_type=="reinit" else False

    # Empty Unless we attacked the dataset
    attack_rate_str = ""

    # Data Loader
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    transform_cifar10= transforms.Compose([
                #transforms.RandomCrop(32, padding=5),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
    transform_cifar10_alexnet = transforms.Compose([
                #transforms.Resize(256),
                #transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # TODO make sure to do correct normalization for each dataset above is MNIST only.

    if args.dataset == "mnist":
        traindataset = datasets.MNIST('../data', train=True, download=True,transform=transform)
        testdataset = datasets.MNIST('../data', train=False, transform=transform)
        '''
        print(type(traindataset.data))
        print(len(traindataset))
        print(traindataset.data.size())
        print(traindataset.data.dtype)
        print(traindataset.targets[0])
        print(type(testdataset))
        print(len(testdataset))
        print(testdataset.targets[0])
        '''
        from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet

    elif args.dataset == "cifar10":
        traindataset = datasets.CIFAR10('../data', train=True, download=True,transform=transform_cifar10)
        testdataset = datasets.CIFAR10('../data', train=False, transform=transform_cifar10)
        from archs.cifar10 import AlexNet, LeNet5, fc1, densenet #,resnet,  vg
        import deeprobust.image.netmodels.resnet as resnet
        import deeprobust.image.netmodels.vgg as vgg

    elif args.dataset == "fashionmnist":
        traindataset = datasets.FashionMNIST('../data', train=True, download=True,transform=transform)
        testdataset = datasets.FashionMNIST('../data', train=False, transform=transform)
        from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet

    elif args.dataset == "cifar100":
        traindataset = datasets.CIFAR100('../data', train=True, download=True,transform=transform)
        testdataset = datasets.CIFAR100('../data', train=False, transform=transform)
        from archs.cifar100 import AlexNet, fc1, LeNet5, vgg, resnet

    # If you want to add extra datasets paste here
    elif args.dataset == "mnist_fgsm_attack":
        attack_rate = args.attack_rate # 50% of the train dataset will be attacked
        attack_rate_str = "_"+str(attack_rate)
        attack_type = 'fgsm'
        attack_dataset = MNIST_AttackedDataset(attack_rate,attack_type)

        traindataset = attack_dataset.create_partial_adverserial_dataset(attack_rate,
                        plot=True, plot_path = f"{os.getcwd()}/plots/attack_samples/{args.dataset}_{attack_type}/")
        testdataset = datasets.MNIST('../data', train=False, transform=transform)

        from archs.mnist_fgsm_attack import AlexNet, LeNet5, fc1, vgg, resnet

    elif args.dataset == "mnist_pgd_attack":
        attack_rate = args.attack_rate # 50% of the train dataset will be attacked
        attack_rate_str = "_"+str(attack_rate)
        attack_type = 'pgd'
        attack_dataset = MNIST_AttackedDataset(attack_rate,attack_type)

        traindataset = attack_dataset.create_partial_adverserial_dataset(attack_rate,
                        plot=True, plot_path = f"{os.getcwd()}/plots/attack_samples/{args.dataset}_{attack_type}/")
        testdataset = datasets.MNIST('../data', train=False, transform=transform)

        from archs.mnist_fgsm_attack import AlexNet, LeNet5, fc1, vgg, resnet


    elif args.dataset == "cifar10_pgd_attack":
        attack_rate = args.attack_rate # 50% of the train dataset will be attacked
        attack_rate_str = "_"+str(attack_rate)
        attack_type ='pgd'
        pgd_attack_dataset = CIFAR10_AttackDataset(attack_rate,attack_type)

        traindataset = pgd_attack_dataset.create_partial_adverserial_dataset(attack_rate, plot=True, recreate=False,
                                plot_path = f"{os.getcwd()}/plots/attack_samples/{args.dataset}_{attack_type}/")

        testdataset = datasets.CIFAR10('../data', train=False, transform=transform_cifar10)

        from archs.cifar10 import AlexNet, LeNet5, fc1, vgg, densenet #,resnet
        import deeprobust.image.netmodels.resnet as resnet
        import deeprobust.image.netmodels.vgg as vgg

    elif args.dataset == "cifar10_fgsm_attack":
        attack_rate = args.attack_rate # 50% of the train dataset will be attacked
        attack_rate_str = "_"+str(attack_rate)
        attack_type ='fgsm'
        fgsm_attack_dataset = CIFAR10_AttackDataset(attack_rate,attack_type)

        traindataset = fgsm_attack_dataset.create_partial_adverserial_dataset(attack_rate, plot=True, recreate=False,
                                plot_path = f"{os.getcwd()}/plots/attack_samples/{args.dataset}_{attack_type}/")

        testdataset = datasets.CIFAR10('../data', train=False, transform=transform_cifar10)

        from archs.cifar10 import AlexNet, LeNet5, fc1, vgg, densenet #,resnet
        import deeprobust.image.netmodels.resnet as resnet
        import deeprobust.image.netmodels.vgg as vgg

    else:
        print("\nWrong Dataset choice \n")
        exit()

    train_loader = torch.utils.data.DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=0,drop_last=False)
    #train_loader = cycle(train_loader)
    test_loader = torch.utils.data.DataLoader(testdataset, batch_size=args.batch_size, shuffle=False, num_workers=0,drop_last=True)

    # Importing Network Architecture
    global model
    if args.arch_type == "fc1":
       model = fc1.fc1().to(device)
    elif args.arch_type == "lenet5":
        model = LeNet5.LeNet5().to(device)
    elif args.arch_type == "alexnet":
        model = AlexNet.AlexNet().to(device)
    elif args.arch_type == "vgg16":
        #model = vgg.vgg16().to(device)
        model = vgg.VGG('VGG16').to(device)
    elif args.arch_type == "resnet18":
        if 'cifar10' in args.dataset:
            model = resnet.ResNet18().to(device)
        else:
            model = resnet.resnet18().to(device)
    elif args.arch_type == "densenet121":
        model = densenet.densenet121().to(device)
    # If you want to add extra model paste here
    else:
        print("\nWrong Model choice\n")
        exit()

    # Weight Initialization
    model.apply(weight_init)

    # Copying and Saving Initial State
    initial_state_dict = copy.deepcopy(model.state_dict())
    utils.checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}{attack_rate_str}/")
    torch.save(model, f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}{attack_rate_str}/initial_state_dict_{args.prune_type}.pth.tar")

    # Making Initial Mask
    make_mask(model)

    # Optimizer and Loss

    #if ('cifar10' in args.dataset) and ('resnet18' == args.arch_type):
    if ('adam' == args.optimizer):
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


    criterion = nn.CrossEntropyLoss() # Default was F.nll_loss

    # Layer Looper
    for name, param in model.named_parameters():
        print(name, param.size())

    # Pruning
    # NOTE First Pruning Iteration is of No Compression
    bestacc = 0.0
    best_accuracy = 0
    ITERATION = args.prune_iterations
    comp = np.zeros(ITERATION,float)
    bestacc = np.zeros(ITERATION,float)
    step = 0
    all_loss = np.zeros(args.end_iter,float)
    all_accuracy = np.zeros(args.end_iter,float)


    for _ite in range(args.start_iter, ITERATION):
        if not _ite == 0:
            prune_by_percentile(args.prune_percent, resample=resample, reinit=reinit)
            if reinit:
                model.apply(weight_init)
                #if args.arch_type == "fc1":
                #    model = fc1.fc1().to(device)
                #elif args.arch_type == "lenet5":
                #    model = LeNet5.LeNet5().to(device)
                #elif args.arch_type == "alexnet":
                #    model = AlexNet.AlexNet().to(device)
                #elif args.arch_type == "vgg16":
                #    model = vgg.vgg16().to(device)
                #elif args.arch_type == "resnet18":
                #    model = resnet.resnet18().to(device)
                #elif args.arch_type == "densenet121":
                #    model = densenet.densenet121().to(device)
                #else:
                #    print("\nWrong Model choice\n")
                #    exit()
                step = 0
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        weight_dev = param.device
                        param.data = torch.from_numpy(param.data.cpu().numpy() * mask[step]).to(weight_dev)
                        step = step + 1
                step = 0
            else:
                original_initialization(mask, initial_state_dict)

            if args.optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


        print(f"\n--- Pruning Level [{ITE}:{_ite}/{ITERATION}]: ---")

        # Print the table of Nonzeros in each layer
        comp1 = utils.print_nonzeros(model)
        comp[_ite] = comp1
        pbar = tqdm(range(args.end_iter))

        for iter_ in pbar:

            # Frequency for Testing
            if iter_ % args.valid_freq == 0:
                accuracy = test(model, test_loader, criterion)

                # Save Weights
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    utils.checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}{attack_rate_str}/")
                    torch.save(model,f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}{attack_rate_str}/{_ite}_model_{args.prune_type}.pth.tar")

            # Training
            loss = train(model, train_loader, optimizer, criterion)
            all_loss[iter_] = loss
            all_accuracy[iter_] = accuracy

            # Frequency for Printing Accuracy and Loss
            if iter_ % args.print_freq == 0:
                pbar.set_description(
                    f'Train Epoch: {iter_}/{args.end_iter} Loss: {loss:.6f} Accuracy: {accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}%')

        writer.add_scalar('Accuracy/test', best_accuracy, comp1)
        bestacc[_ite]=best_accuracy

        # Plotting Loss (Training), Accuracy (Testing), Iteration Curve
        #NOTE Loss is computed for every iteration while Accuracy is computed only for every {args.valid_freq} iterations. Therefore Accuracy saved is constant during the uncomputed iterations.
        #NOTE Normalized the accuracy to [0,100] for ease of plotting.
        plt.plot(np.arange(1,(args.end_iter)+1), 100*(all_loss - np.min(all_loss))/np.ptp(all_loss).astype(float), c="blue", label="Loss")
        plt.plot(np.arange(1,(args.end_iter)+1), all_accuracy, c="red", label="Accuracy")
        plt.title(f"Loss Vs Accuracy Vs Iterations ({args.dataset},{args.arch_type})")
        plt.xlabel("Iterations")
        plt.ylabel("Loss and Accuracy")
        plt.legend()
        plt.grid(color="gray")
        utils.checkdir(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}{attack_rate_str}/")
        plt.savefig(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}{attack_rate_str}/{args.prune_type}_LossVsAccuracy_{comp1}.png", dpi=1200, bbox_inches="tight")
        plt.close()

        # Dump Plot values
        utils.checkdir(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}{attack_rate_str}/")
        all_loss.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}{attack_rate_str}/{args.prune_type}_all_loss_{comp1}.dat")
        all_accuracy.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}{attack_rate_str}/{args.prune_type}_all_accuracy_{comp1}.dat")

        # Dumping mask
        utils.checkdir(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}{attack_rate_str}/")
        with open(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}{attack_rate_str}/{args.prune_type}_mask_{comp1}.pkl", 'wb') as fp:
            pickle.dump(mask, fp)

        # Making variables into 0
        best_accuracy = 0
        all_loss = np.zeros(args.end_iter,float)
        all_accuracy = np.zeros(args.end_iter,float)

    # Dumping Values for Plotting
    utils.checkdir(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}{attack_rate_str}/")
    comp.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}{attack_rate_str}/{args.prune_type}_compression.dat")
    bestacc.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}{attack_rate_str}/{args.prune_type}_bestaccuracy.dat")

    # Plotting
    a = np.arange(args.prune_iterations)
    plt.plot(a, bestacc, c="blue", label="Winning tickets")
    plt.title(f"Test Accuracy vs Unpruned Weights Percentage ({args.dataset},{args.arch_type})")
    plt.xlabel("Unpruned Weights Percentage")
    plt.ylabel("test accuracy")
    plt.xticks(a, comp, rotation ="vertical")
    plt.ylim(0,100)
    plt.legend()
    plt.grid(color="gray")
    utils.checkdir(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}{attack_rate_str}/")
    plt.savefig(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}{attack_rate_str}/{args.prune_type}_AccuracyVsWeights.png", dpi=1200, bbox_inches="tight")
    plt.close()

# Function for Training
def train(model, train_loader, optimizer, criterion):
    EPS = 1e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for batch_idx, (imgs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        #imgs, targets = next(train_loader)
        imgs, targets = imgs.to(device), targets.to(device)
        output = model(imgs)
        train_loss = criterion(output, targets)
        train_loss.backward()

        # Freezing Pruned weights by making their gradients Zero
        for name, p in model.named_parameters():
            if 'weight' in name:
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor < EPS, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)
        optimizer.step()
    return train_loss.item()

# Function for Testing
def test(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy

# Prune by Percentile module
def prune_by_percentile(percent, resample=False, reinit=False,**kwargs):
        global step
        global mask
        global model

        # Calculate percentile value
        step = 0
        for name, param in model.named_parameters():

            # We do not prune bias term
            if 'weight' in name:
                tensor = param.data.cpu().numpy()
                alive = tensor[np.nonzero(tensor)] # flattened array of nonzero values
                percentile_value = np.percentile(abs(alive), percent)

                # Convert Tensors to numpy and calculate
                weight_dev = param.device
                new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])

                # Apply new weight and mask
                param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
                mask[step] = new_mask
                step += 1
        step = 0

# Function to make an empty mask of the same size as the model
def make_mask(model):
    global step
    global mask
    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            step = step + 1
    mask = [None]* step
    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            mask[step] = np.ones_like(tensor)
            step = step + 1
    step = 0

def original_initialization(mask_temp, initial_state_dict):
    global model

    step = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            weight_dev = param.device
            param.data = torch.from_numpy(mask_temp[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            step = step + 1
        if "bias" in name:
            param.data = initial_state_dict[name]
    step = 0

# Function for Initialization
def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


if __name__=="__main__":

    #from gooey import Gooey
    #@Gooey

    # Arguement Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",default= 1.2e-3, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=60, type=int)
    parser.add_argument("--start_iter", default=0, type=int)
    parser.add_argument("--end_iter", default=100, type=int)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--valid_freq", default=1, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prune_type", default="lt", type=str, help="lt | reinit")
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--dataset", default="mnist", type=str, help="mnist | cifar10 | fashionmnist | cifar100")
    parser.add_argument("--arch_type", default="fc1", type=str, help="fc1 | lenet5 | alexnet | vgg16 | resnet18 | densenet121")
    parser.add_argument("--prune_percent", default=10, type=int, help="Pruning percent")
    parser.add_argument("--prune_iterations", default=35, type=int, help="Pruning iterations count")
    parser.add_argument("--attack_rate", default=10, type=int, help="Attack rate as percentage")
    parser.add_argument("--optimizer", default='sgd', type=str, help="Optimizer type")
    parser.add_argument("--momentum", default=0.5, type=float, help="Momentum")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight Decay")


    args = parser.parse_args()


    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu


    #FIXME resample
    resample = False

    # Looping Entire process
    #for i in range(0, 5):
    main(args, ITE=1)
