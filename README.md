# GaussianPhylogeny

## Generating data
The code for this is in `OT-Tree.jl`. 
First, create an array of `OTnode` objects by reading the tree topology file. Example of such a file can be found in `testing/tree_test.txt`. The information is in thee columns: index of node in the tree, index of parent node, and time of the branch separating given node from parent. This information is enough to reconstruct the topology. 
Then, sample all the nodes in the tree using `SampleTree!` function. 

An example:  
`OTgraph = ReadTree("testing/tree_test.txt")`  
`C = [1 0.5 ; 0.5 1]`  
`SampleTree!(OTgraph, C)`  
Since the correlation matrix is 2x2 in this example, the configurations will be gaussian vectors of length 2. 

## Computing likelihood of data
To compute the likelihood of data given some trial correlation matrix `K` and time on branches `dt`, one uses the `ComputeLikelihood` function. This function itself calls the message passing algorithm, and performs the necessary gaussian integrations. 
For example  
`a = ComputeLikelihood(OTgraph, K, dt)`  
and a\[1\] will contain the likelihood of data contained in `OTgraph` leaves. 