# Adversarial Julia

## Inventory 
_Adversarial.jl_: the file containing the adversarial module  
_Demo.ipynb_: a notebook to demonstrate how the module works  
_Midterm Progress.ipynb_: A notebook that includes legacy results. The first few cells deal with generating adversarial examples against a malware classifier  
_Project Presentation.pptx_: The presentation I gave in class  

## Required Packages
Distances, Knet, StatsBase, Optim, AutoGrad, ProgressMeter, Statistics, ForwardDiff, IterTools

## Running the Code
See the "Demo" notebook for examples. In general, first load the data, initialize a classifier (this can be done with the built in adv.MLP struct), and train it (with adv.trainresults). Once trained, you can run an attack on the model using one of the attack functions defined in Adversarial.jl. Use adv.targetted_attack and adv_untargetted_attack to perform an attack over all input  

### Malware Classification Attack
This code can be found in the first several cells of Midterm Progress.ipynb