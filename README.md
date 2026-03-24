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


## Examples


A simple example with KMedoidsQuery is provided in [notebooks/kmedoids_toy_data.ipynb](notebooks/kmedoids_toy_data.ipynb).

An more comprehensive example is provided in [notebooks/Superconductivity.ipynb](notebooks/Superconductivity.ipynb), which demonstrates every method on the [superconductivity](https://archive.ics.uci.edu/dataset/464/superconductivty+data) benchmark dataset.
