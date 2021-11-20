import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from tqdm import tqdm
import utils

DPI = 1200
prune_iterations = 7 #35
arch_types = ["fc1"] #, "lenet5", "resnet18"]
datasets = ["mnist_fgsm_attack"] #"fashionmnist", "cifar10", "cifar100"]
attack_rates = [*range(10,44,10)] #[*range(10,99,10)]
colors = ["blue", "green", "red", "orange"]#, "yellow", "cyan", "magenta",  "darkviolet", "lime"]

for arch_type in tqdm(arch_types):
    for ind, dataset in enumerate(tqdm(datasets)):
        for attack_rate, color in tqdm(zip(attack_rates,colors)):
            attack_rate_str = '_'+str(attack_rate)
            d = np.load(f"{os.getcwd()}/dumps/lt/{arch_type}/{dataset}{attack_rate_str}/lt_compression.dat", allow_pickle=True)
            b = np.load(f"{os.getcwd()}/dumps/lt/{arch_type}/{dataset}{attack_rate_str}/lt_bestaccuracy.dat", allow_pickle=True)
            #c = np.load(f"{os.getcwd()}/dumps/lt/{arch_type}/{dataset}{attack_rate_str}/reinit_bestaccuracy.dat", allow_pickle=True)

            #plt.clf()
            #sns.set_style('darkgrid')
            #plt.style.use('seaborn-darkgrid')
            a = np.arange(prune_iterations)
            plt.plot(a, b, c=color, label=f"{attack_rate}%")
            #plt.plot(a, c, c="red", label="Random reinit")
        plt.title(f"Test Accuracy vs Weights % ({arch_type} | {dataset})")
        plt.xlabel("Remaining Weights %")
        plt.ylabel("Test accuracy")
        plt.xticks(a, d, rotation ="vertical")
        plt.ylim(0,100)
        plt.legend(title='Attack Rate')
        #ax.legend()
        plt.grid(color="gray")

        utils.checkdir(f"{os.getcwd()}/plots/lt/combined_plots/experiment1/")
        plt.savefig(f"{os.getcwd()}/plots/lt/combined_plots/experiment1/combined_{arch_type}_{dataset}.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        #print(f"\n combined_{arch_type}_{dataset} plotted!\n")