# The (Un)Scalability of Informed Heuristic Function Estimation in NP-Hard Search Problems

Official Repository for the paper titled "[The (Un)Scalability of Informed Heuristic Function Estimation in NP-Hard Search Problems]([https://openreview.net/forum?id=33wyZ4xTIx](https://openreview.net/pdf?id=JllRdycmLk))"


# Setup
```
conda create --name <env> --file <this file>
```
Install [pytorch](https://pytorch.org/get-started/locally/)


# Run 
Run as 
```
python train.py -d pancake -n 15 -m 4 -o pancake/set_2 -i 1000000
```

More information about the command line parameters can be obtained as
```
python train.py --help
```
Experiments for different loss threshold can be obtained by changing threshold on line 19

Experiments on fixed depth can be obtained by changing on line 360
```
replace search_width search_depth 
```

# Citing
If you found our repository/experiments useful please consider citing our work as:

```
@article{
pendurkar2023the,
title={The (Un)Scalability of Informed Heuristic Function Estimation in {NP}-Hard Search Problems},
author={Sumedh Pendurkar and Taoan Huang and Brendan Juba and Jiapeng Zhang and Sven Koenig and Guni Sharon},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2023},
url={https://openreview.net/forum?id=JllRdycmLk},
note={}
}
```
