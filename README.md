# Lottery Ticket Hypothesis on Adversarial Datasets

cs269-lsml-project

[Original LTH code reference](https://github.com/rahulvigneswaran/Lottery-Ticket-Hypothesis-in-Pytorch)


[Adversary examples created using DeepRobust library](https://github.com/DSE-MSU/DeepRobust)

### Create Environment

Run the following commands to create the environment:
```
conda create --name lsml python=3.6

conda activate lsml

pip install -r requirements2.txt

```

Run the LTH code:

```
python3 main.py --prune_type=lt --arch_type=fc1 --dataset=mnist_fgsm_attack --prune_percent=90 --prune_iterations=2 --end_iter=3

```

You can modify the parameters based on  [Original LTH code reference](https://github.com/rahulvigneswaran/Lottery-Ticket-Hypothesis-in-Pytorch#readme)

__Global numbers to run on experiments:__

We can use the following parameters as fixed in our experiments and set up the remaining:
__MNIST__
```
--end_iter=15  --prune_percent=30 --prune_iterations=15  --batch_size=256 --lr=0.01
```
__CIFAR10__
```
--end_iter=15  --prune_percent=30 --prune_iterations=15  --batch_size=128 --lr=0.1
```

Set different values based on the experiment you are running:
```
--prune_type --arch_type --dataset --attack_rate
```
Also make sure to run same experiment with both --prune_type=lt and --prune_type=reinit.


### Research Questions:
1. Can we find winning tickets with attacked dataset?
2. Are winning tickets same for org. & attacked dataset? If diff by how much? (Experiment 3)
3. How attack rate on dataset changes LTH results? (Experiment 1)
4. Different attacks have diff. effects on LTH results? ( Experiment 2)
5. Different type of architectures on same attacked dataset affected differently from the attacks or have same LTH results/accuracy? (Experiment 4)
6. Can we see similar results on different Datasets (CIFAR10) - (Experiment 5)


### Experiments :

1. Use Original + Generated Dataset (Compare performance for dataset ratios)
 a. 50% original MNIST + 50 % adversarial samples

2. Check different attack types and how they are behaving in terms of LTH:
example attacks (2-3) [FGSM, PGD, CW]

3. Compare winning tickets generated by original vs generated dataset
write code to read pruned networks and compare against each other, check how much they are similar? ie. also define similarity metric.

4. Different architectures [VGG,ResNet..]

5. Different datasets (CIFAR10)
6. Oneshot pruning vs iterative pruning experiment.
