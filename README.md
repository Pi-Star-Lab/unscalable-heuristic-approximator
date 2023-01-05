# The (Un)Scalability of Heuristic Approximators for NP-Hard Search Problems

Official Repository for "[(Un)Scalability of Heuristic Approximators for NP-Hard Search Problems](https://openreview.net/forum?id=33wyZ4xTIx)"


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
  @inproceedings{pendurkar2022nphsearch,
      title={The (Un)Scalability of Heuristic Approximators for NP-Hard Search Problems},
      author={Pendurkar, Sumedh and Huang, Taoan and Koenig, Sven and Sharon, Guni},
      booktitle={In I (Still) Can't Believe It's Not Better! NeurIPS 2022 Workshop},
      volume={},
      number={},
      pages={},
      year={2022}
  }
```
