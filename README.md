# Scalability of learning heuristic functions

Setup the environment
```
conda create --name <env> --file <this file>
```
Install [pytorch](https://pytorch.org/get-started/locally/)


Run the code as
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

Note: Dataset for TSP is not provided due to size constraints

nevertheless dataset can be generated as
```
python generate_dataset.py -d tsp -n 12 -m 3 -o set -i 100000 
```
