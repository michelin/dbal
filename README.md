# Discrepancy-Based Active Learning for Domain Adaptation

This library offers several query methods for active learning, in the context of domain-expansion tasks :
- KMedoidsQuery: based on the K-medoids clustering algorithm;
- KMeansQuery: based on the K-means clustering algorithm;
- KCenterQuery: based on the k-centers algorithm;
- DiversityQuery: select target points with maximum average-distance to source points;
- RandomQuery: A baseline method that randomly selects samples from the target domain. To be used only in comparison with other methods.
- OrderedQuery: a basic method that should be used in conjonction with uncertainty predictions provided by the following approaches:
    - AADA: an hybrid active learning method for domain adaptation using a combination of entropy measure from a classifier and the outputs of a domain discriminator;
    - QBC: Query By Commitee;
    - BVSB: Best versus second best.;


## Example


A simple example with KMedoidsQuery is provided in [notebooks/kmedoids_toy_data.ipynb](notebooks/kmedoids_toy_data.ipynb).


## Experiments

The experiments are conducted on three benchmark datasets:
- Superconductivity [UCI](https://archive.ics.uci.edu/ml/datasets/superconductivty+data#)
- Office [Berkeley](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/#datasets_code)
- Digits [SYNTH](http://yaroslav.ganin.net/),  [SVHN](http://ufldl.stanford.edu/housenumbers/)

Experiments can be run with the following command lines:

```
python scripts/run_experiments.py
```

## Notebooks

Quick results can be obtained in the `notebooks` folder:

### Superconductivity 

[![name](images/superconductivity.png)](https://github.com/AnonymousAccount0/dbal/blob/master/notebooks/Superconductivity.ipynb)

### Office

[![name](images/office.png)](https://github.com/AnonymousAccount0/dbal/blob/master/notebooks/Office.ipynb)

### Digits

[![name](images/digits.png)](https://github.com/AnonymousAccount0/dbal/blob/master/notebooks/Digits.ipynb)

