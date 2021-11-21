import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,models,transforms
from PIL import Image

from deeprobust.image.attack.pgd import PGD
import deeprobust.image.netmodels.resnet as resnet
import deeprobust.image.netmodels.CNN as CNN
from deeprobust.image.config import attack_params
import matplotlib.pyplot as plt
import os.path
import deeprobust.image.netmodels.train_model as trainmodel
from deeprobust.image.utils import download_model

model = resnet.ResNet18().to('cpu')
print("Load network")

#import ipdb
#ipdb.set_trace()


file_path = './trained_models/CIFAR10_ResNet18_epoch_20.pt'
if not os.path.exists(file_path):
    #trainmodel.train('ResNet18', 'CIFAR10', 'cpu', 20)
    URL = "https://github.com/I-am-Bot/deeprobust_model/raw/master/CIFAR10_ResNet18_epoch_20.pt"
    download_model(URL, file_path)

model.load_state_dict(torch.load("./trained_models/CIFAR10_ResNet18_epoch_20.pt",map_location=torch.device('cpu')))
model.eval()

transform_val = transforms.Compose([
                transforms.ToTensor(),
                ])
#transform=transforms.Compose([transforms.ToTensor()])

#train_data = datasets.CIFAR10('../data', download = True, train= True, transform=transform)
#print(type(train_data))
#print(len(train_data))
#print(train_data)
#print(train_data.targets.size())

#xx = train_data.data[:100].to('cpu')

#print(xx.size())

## Set Target
#yy = train_data.targets[:100].to('cpu')

test_loader  = torch.utils.data.DataLoader(
                datasets.CIFAR10('../data', train = False, download=True,
                transform = transform_val),
                batch_size = 60, shuffle=True) #, **kwargs)


classes = np.array(('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'))

xx, yy = next(iter(test_loader))
xx = xx.to('cpu').float()

predict0 = model(xx)
predict0= predict0.argmax(dim=1, keepdim=True)

adversary = PGD(model, device = 'cpu')
AdvExArray = adversary.generate(xx, yy, **attack_params['PGD_CIFAR10']).float()

predict1 = model(AdvExArray)
predict1= predict1.argmax(dim=1, keepdim=True)

print('====== RESULT =====')
print('true label',classes[yy.cpu()])
print('predict_orig',classes[predict0.cpu()])
print('predict_adv',classes[predict1.cpu()])

x_show = xx.cpu().numpy().swapaxes(1,3).swapaxes(1,2)[0]
# print('xx:', x_show)
plt.imshow(x_show, vmin = 0, vmax = 255)
plt.savefig('./adversary_examples/cifar_advexample_orig.png')
# print('x_show', x_show)


# print('---------------------')
AdvExArray = AdvExArray.cpu().detach().numpy()
AdvExArray = AdvExArray.swapaxes(1,3).swapaxes(1,2)[0]

# print('Adv', AdvExArray)

# print('----------------------')
# print(AdvExArray)
plt.imshow(AdvExArray, vmin = 0, vmax = 255)
plt.savefig('./adversary_examples/cifar_advexample_pgd.png')
