# Gradient Matching for Domain Generalisation
<img src="https://user-images.githubusercontent.com/18204038/125197696-a9470200-e256-11eb-98e7-66123e34b59f.png" width="350" align="right">

This is the official PyTorch implementation of [Gradient Matching for Domain Generalisation](https://arxiv.org/abs/2104.09937). In our paper, we propose an *inter-domain gradient matching* (IDGM) objective that targets domain generalization by maximizing the inner product between gradients from different domains.
To avoid computing the expensive second-order derivative of the IDGM objective, we derive a simpler first-order algorithm named **Fish** that approximates its optimization. 

This repository contains code to reproduce the main results of our paper.



## Dependencies

**(Recommended)** You can setup up conda environment with all required dependencies using `environment.yml`:
```shell
conda env create -f environment.yml
conda activate fish
```

Otherwise you can also install the following packages manually:
```
python=3.7.10
numpy=1.20.2
pytorch=1.8.1
torchaudio=0.8.1
torchvision=0.9.1
torch-cluster=1.5.9
torch-geometric=1.7.0
torch-scatter=2.0.6
torch-sparse=0.6.9
wilds=1.1.0
scikit-learn=0.24.2
scipy=1.6.3
seaborn=0.11.1
tqdm=4.61.0
```



## Running Experiments
We offer options to train using our proposed method Fish or by using Empirical Risk Minimisation baseline. 
This can be specified by the `--algorithm` flag (either `fish` or `erm`).


#### CdSprites-N

We propose this simple shape-color dataset based on the [dSprites](https://github.com/deepmind/dsprites-dataset) dataset,
which contains a collection of white 2D sprites of different shapes, scales, rotations and positions.
The dataset contains `N` domains, where `N` can be specified. The goal is to classify the shape of the sprites, 
and there is a shape-color deterministic matching that is specific per domain.
This way we have shape as the invariant feature and color as the spurious feature. 
On the test set, however, this correlation between color and shape is removed.
See the image below for an illustration.
<p align='center'><img src="https://user-images.githubusercontent.com/18204038/125146851-ff655980-e11f-11eb-8086-f51ad55f3bc5.png" alt="cdsprites" width="300"/></p>

The CdSprites-N dataset can be downloaded [here](https://yugeten.github.io/files/cdsprites.zip). 
After downloading, please extract the zip file to your preferred data dir (e.g. `<your_data_dir>/cdsprites`). 
The following command runs an experiment using Fish with number of domains `N=15`:

```shell
python main.py --dataset cdsprites --algorithm fish --data-dir <your_data_dir> --num-domains 15
```

The number of domains you can choose from are: `N = 5, 10, 15, 20, 25, 30, 35, 40, 45, 50`. 

#### WILDS
We include the following 6 datasets from the WILDS benchmark:
[`amazon`](https://wilds.stanford.edu/datasets/#amazon),
[`camelyon`](https://wilds.stanford.edu/datasets/#camelyon17),
[`civil`](https://wilds.stanford.edu/datasets/#civilcomments),
[`fmow`](https://wilds.stanford.edu/datasets/#fmow),
[`iwildcam`](https://wilds.stanford.edu/datasets/#iwildcam),
[`poverty`](https://wilds.stanford.edu/datasets/#povertymap).
The datasets can be downloaded automatically to a specified data folder. For instance, to train with Fish on Amazon dataset, simply run:
```shell
python main.py --dataset amazon --algorithm fish --data-dir <your_data_dir>
```
This should automatically download the Amazon dataset to `<your_data_dir>/wilds`. Experiments on other datasets can be ran by the following commands:
```shell
python main.py --dataset camelyon --algorithm fish --data-dir <your_data_dir>
python main.py --dataset civil --algorithm fish --data-dir <your_data_dir>
python main.py --dataset fmow --algorithm fish --data-dir <your_data_dir>
python main.py --dataset iwildcam --algorithm fish --data-dir <your_data_dir>
python main.py --dataset poverty --algorithm fish --data-dir <your_data_dir>
```

Alternatively, you can also download the datasets to `<your_data_dir>/wilds` manually by following the instructions [here](https://wilds.stanford.edu/get_started/). See current results on WILDS here:
![image](https://user-images.githubusercontent.com/18204038/125197885-68032200-e257-11eb-9521-2331f7907816.png)

#### DomainBed
For experiments on datasets including CMNIST, RMNIST, VLCS, PACS, OfficeHome, TerraInc and DomainNet,
we implemented Fish on the [DomainBed](https://arxiv.org/abs/2007.01434) benchmark (see [here](https://github.com/facebookresearch/DomainBed)) 
and you can compare our algorithm against up to 20 SOTA baselines. See current results on DomainBed here:

![image](https://user-images.githubusercontent.com/18204038/125197874-591c6f80-e257-11eb-9061-4ecca7065978.png)


## Citation
If you make use of this code in your research, we would appreciate if you considered citing the paper that is most relevant to your work:
```
@article{shi2021gradient,
	title="Gradient Matching for Domain Generalization.",
	author="Yuge {Shi} and Jeffrey {Seely} and Philip H. S. {Torr} and N. {Siddharth} and Awni {Hannun} and Nicolas {Usunier} and Gabriel {Synnaeve}",
	journal="arXiv preprint arXiv:2104.09937",
	year="2021"}
```



## Contributions

We welcome contributions via pull requests. Please email yshi@robots.ox.ac.uk or gab@fb.com for any question/request.
