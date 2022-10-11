# VAESA Dataset

The repo contains the training data for the VAESA paper. We store them in the `.csv` format under `db` directory. 

The data are collected using random search, grid search, and gradient descent. 
There are three subfolders in the `db` folder: 
1. `layer_db` contains the performance of different architecture configurations running different convolution layer configurations. There are data columns in the csv to specify the layer dimensions.
2. `network_db` contains the performance of different architecture configurations running different network configurations. The latency and energy reported are the sum of all layers in the network.  
3. `all_db` constains the performance of different architecture configurations running all four target networks (alexnet, resnet, resnext, deepbench). 

The data we used in the VAE training is the consolidated csv stored in `all_layers_db/dataset_all_layer_large.csv`.
Note that we use CoSA to generate the mapping for different problem and architecture configuration file. The mapping features are not currently logged in the csv files. 
