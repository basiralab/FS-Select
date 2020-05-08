# Identifying the best data-driven feature selection method for boosting reproducibility in classification tasks

This repository contains FS-Select Python source code to learn how to identify the most reproducible feature selection method for a given dataset. This has been re-coded up by Raouaa Ghodhbani (raouaghodhbani@gmail.com).

```

██████╗  █████╗ ███████╗██╗██████╗  █████╗     ██╗      █████╗ ██████╗ 
██╔══██╗██╔══██╗██╔════╝██║██╔══██╗██╔══██╗    ██║     ██╔══██╗██╔══██╗
██████╔╝███████║███████╗██║██████╔╝███████║    ██║     ███████║██████╔╝
██╔══██╗██╔══██║╚════██║██║██╔══██╗██╔══██║    ██║     ██╔══██║██╔══██╗
██████╔╝██║  ██║███████║██║██║  ██║██║  ██║    ███████╗██║  ██║██████╔╝
╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝╚═╝  ╚═╝╚═╝  ╚═╝    ╚══════╝╚═╝  ╚═╝╚═════╝ 
                                                                       

Developed by Raouaa Ghodhbani © | 2020
raouaghodhbani@gmail.com
```

## Motivation: how much can you trust your FS method? 

Considering the proliferation of extremely high-dimensional data in many domains including computer vision and healthcare applications such as computer-aided diagnosis (CAD), advanced techniques for reducing the data dimensionality and identifying the most relevant features for a given classification task such as distinguishing between healthy and disordered brain states are needed. Despite the existence of many works on boosting the classification accuracy using a particular feature selection (FS) method, *choosing the best one from a large pool of existing FS techniques for boosting feature reproducibility within a dataset of interest remains a formidable challenge to tackle.* Notably, a good performance of a particular FS method does not necessarily imply that the experiment is reproducible and that the features identified are optimal for the entirety of the samples. Essentially, this paper presents the first attempt to address the following challenge: **"Given a set of different feature selection methods {FS1, ..., FSK }, and a dataset of interest, how to identify the most reproducible and 'trustworthy' connectomic features that would produce reliable biomarkers capable of accurately differentiate between two specific conditions?"** 

To this aim, we propose FS-Select framework which explores the relationships among the different FS methods using a multi-graph architecture based on feature reproducibility power, average accuracy, and feature stability of each FS method. By extracting the 'central' graph node, we identify the most reliable and reproducible FS method for the target brain state classification task along with the most discriminative features fingerprinting these brain states. To evaluate the reproducibility power of FS-Select, we perturbed the training set by using different cross-validation strategies on a multi-view small-scale connectomic dataset (late mild cognitive impairment vs Alzheimer's disease) and large-scale dataset including autistic vs healthy subjects. Our experiments revealed reproducible connectional features fingerprinting disordered brain states.

![BrainNet pipeline](https://github.com/basiralab/FS-Select/blob/master/Fig1.png)


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

Libraries you need to install to run the demo.

```
Python 2.7 and Python 3
numPy
sciPy
scikit-learn
scikit-feature
matplotlib
seaborn
networkx
pickle
pandas
hypernetx
sklearn
```


## Demo
Run 
```
run_demo.ipynb
```
## FS-Select YouTube video

To learn about how FS-Select works, check the following YouTube video:

https://www.youtube.com/watch?v=9HbLxNef2t8

## Citations
## Please cite the following papers when using FS-Select

```
@article{georges2020,
  title={Identifying the best data-driven feature selection method for boosting reproducibility in classification tasks},
  author={Georges, Nicolas and Mhiri, Islem and Rekik, Islem and Alzheimer’s Disease Neuroimaging Initiative and others},
  journal={Pattern Recognition},
  pages={107183},
  year={2020},
  publisher={Elsevier}
}

@inproceedings{georges2018data,
  title={Data-specific feature selection method identification for most reproducible connectomic feature discovery fingerprinting brain states},
  author={Georges, Nicolas and Rekik, Islem and others},<br/>
  booktitle={International Workshop on Connectomics in Neuroimaging},
  pages={99--106},
  year={2018},
  organization={Springer}
}

```

Published paper link: https://www.sciencedirect.com/science/article/pii/S0031320319304832

Paper preprint link on ResearchGate:
https://www.researchgate.net/publication/338147218_Identifying_the_best_data-driven_feature_selection_method_for_boosting_reproducibility_in_classification_tasks


## Acknowledgments

* My Deepest gratitude goes to Dr. Islem Rekik for this outstanding experience. I learnt a lot from you and I am still learning. Thank you for all the knowledge, patience, support and encouragement.


