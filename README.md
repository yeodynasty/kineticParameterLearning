# kineticParameterLearning
1. This is a set of scripts used for recovering reaction kinetic parameters of biochemical pathways based on the work of Yeo et al. 2024: 'Identifying effective evolutionary strategies-based protocol for uncovering reaction kinetic parameters under the effect of measurement noises' https://www.biorxiv.org/content/10.1101/2024.03.05.583637v1.abstract

2. Notebooks for estimating the parameter values of various rate law formulations are in the following folders:
i. Convenience
ii. GMA
iii. Linlog
iv. MM

3. Notebooks for evaluating the effects of datapoint spacing & data augmentation are also in the subfolders of: 
i. GMA
ii. MM

4. Script for exploring the accuracy of average kinetic parameter estimate based on multiple initial seed solutions are in this folder: 'Multiple seeds'. 

5. The synthetic data generated from various rate law formulations for the purpose of parameter estimation:
i. Data.zip

6. Miscellaneous notebooks for exploring the effect of various SavlgoFilter parameter values are in this folder: 'OtherSalvgoFilter/' 
