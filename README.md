# Protein Patch GNNs

## Who?

- Kannan Sankar
- Yang Yang
- Xiao, Mei
- Ma, Eric
- Jiayi Cox
- Barbara Brannetti

To move nimbly, we probably should to keep the team small.
Minimize coordination time, maximize code production time.

## But why??

This is a high risk, potentially high reward project,
in which we aim to develop a graph neural network
capable of doing learning on protein structures.

The end product "goal" here is to produce
a pre-trained graph neural network model of protein structure _patches_,
using a graph representation of protein structure as the input.
This would be in the same mould as the UniRep model,
except done for protein structures rather than protein sequences.

The derivative goals here are as follows:

- Educational: Level-up colleagues' interest and knowledge level on graphs, machine learning, and their connection.
- Educational: Spread "good practices" from the software development world into the data science world.
- Intellectual: Organize knowledge surrounding graph theory and machine learning.
- Practical: Aspiration towards a project codebase that is fully reproducible and continuously tested and analyzed. (Helps our DevOps team in NX.)
- Professional/Community: Produce an artifact of some kind (paper, software library) that gets released to the world.

This is high risk because there could be a big chance no "positive results"
(in the sense of machine learning results) come out of this effort.
There are ways to de-risk this, which I will address later.
However, the rewards could be plentiful
if this turns out to be a great model class to work with.

## Where's this even going to be useful?

It'll take a moment to explain the connection,
but let's spill the beans first on where we think this model could be useful:

(with contributions from everyone on the original email thread)

- Helping with prediction of whether a particular amino acid will contain a liability (think Met oxidation).
- Predicting whether a molecule will bind to a particular patch of a protein.
- Identifying which part of a protein will serve as a binding site for other proteins.
- Use them for protein-protein interaction “prediction and engineering”,
    - imagine to use the patches as residue-residue frequency pair matrices are used to describe interfaces, the patches would contain more information on the interfaces
- Patches could be used to search for similar ones and we might use them to transfer functional sites from one protein to another or to engineer them to change the function
- Might help designing antibodies de novo!

## OK, show me the science!

### Proteins as graphs

First off, let's talk how protein structures can be represented as graphs.
One useful representation amongst many is to consider each amino acid
in a protein structure as a "node" in a graph.
The idea generalizes nicely for dimers/multimers
as long as we have a way of uniquely identifying each amino acid on each X-mer,
but the core idea here is that each node is an amino acid.

How then do we define "edges" in a graph? Here's a few proposals:

- Edges can be defined purely by Angstrom distance from one amino acid C-alpha to another. This is possibly too simplistic to be useful.
- Edges can be defined by biochemical interactions, such as the kind we learn in 3rd year undergrad biochemistry (or high school biology if you studied in Asia). These are the hydrophobic interactions, hydrogen bonds, ionic interactions, and covalent bonds (disulphide bridges and amino acid backbones). In a Python package I developed in grad school, called protein-interaction-network on GitHub, we construct a graph based on all of those interaction terms, with the bonds being inferred by angstrom distance between the relevant atoms in a protein.
- There's also an idea called "Delaunay triangulation", it's related to constructing n-dimensional triangles (tetrahedrons in protein structure). Edges exist between the nodes that form a tetrahedron, with a tetrahedron defined by the three nearest points to a given point.

### How do graphs relate to neural networks?

This goes deep! Here's how.

Every graph is an "object" comprising of two entities: a node set and an edge set.

Each node in the node set can have a vector representation.
For example, we can have a three-slot vector for a carbon atom,
such that it records (6, 4, 1),
with "6" being the atomic number of the atom,
"4" being the number of valence electrons,
and "1" being an indicator variable that states
whether the carbon is the alpha carbon or not.

If we stack up each node's vectors, we get a "node feature matrix".
The order doesn't matter... until we consider the adjacency matrix.

The adjacency matrix canonically is a square matrix of shape (num_nodes, num_nodes).
Canonically, if the graph is undirected,
that is, the edges have no inherent directional meaning,
then it is a symmetric square matrix.
The order in which the nodes are aligned across the matrix matters,
and canonically it is ordered identically along rows and columns
to produce a matrix diagonal that indexes into each node in order.
Edges usually are represented with "1"s,
though weights (if they exist) can be provided too.

If we wanted to generalize the adjacency matrix
such that we are interested in edge types being represented
independent of the others,
then we can instantiate a 3D tensor that is of shape
(num_nodes, num_nodes, num_edge_types).

Other adjacency-like matrices also exist,
such as the graph laplacian matrix (which is like a graph derivative of sorts).

In graph deep learning, we divide tasks into the following categories:

- Node property prediction
- Edge property prediction
- Whole graph property prediction

Graph neural networks take in the node feature matrix,
and dot product them against the adjacency matrix
to perform a step called "Message Passing".
Colloquially, node features are aggregated across neighbors.
Mathematically, this is just a dot product of adjacency matrix with feature matrix.
Then, another dot product transformation against learned weights happens for the feature matrix.
(We have assumed deep learning knowledge in this step,
so please reach out if it is unclear).
In a later step, node information is aggregated,
producing a fixed-length vector representation - one per graph.
And finally, we do further "deep learning" operations
(they're nothing more than matrix multiplies!)
until we predict the "target" we are trying to predict.

### What about protein patches?

Now, imagine if we were to create one graph per protein/protein complex.

A protein patch _can_ be defined as a node + its neighbors
up to N degrees of separation outwards.
(This is a modelling choice that we will have to worry about.)
The semantics of this have to be well-defined, by the way,
so for example, we could do a protein patch
that is 3 degrees of separation across the entire structure,
or we can do 3 degrees of separation
while factoring in only amino acids exposed on the surface.

There are probably other ways to generate protein patches,
perhaps involving fancy graph cutting algorithms,
but for now,
focusing on a node and walking out N degrees of separation
is the most parsimonious way we could go about doing this.

### Sounds cool! What do we have to do, then?

Machine learning projects are best structured as pipelines
(which themselves are graphs, oh-so-meta!).
In terms of an "idealized" pipeline, we would want to build the following:

- Automated homology modelling: From sequence string to PDB file. (Allows us to augment data where not available.)
- Automated graph construction: From PDB file to graph object. An MVP exists for this in the protein-interaction-network package, but could do with better software work. Uses the NetworkX package.
- Automated graph patch generation: From graph object to subgraphs that contain protein patch graphs. This is something that just needs code written and tested, as long as it works with the NetworkX package we're gold!
- Graph neural network model package: Takes in protein graph patches, and produces a fixed-length vector per graph that can be used in an end-to-end differentiable neural network model. Will include code to convert protein patch graph object into features and adjacency-like matrices.
- Defined pretext learning task: An untrained neural network provides the equivalent of a "random projection". That can be handy, but supposedly, pre-trained networks are better. I think a pre-trained task we could do is predicting one of the properties of a node, such as its identity, or its valence, or its pKa, or something else. There are very, very efficient ways to write this code using JAX, and it's something worth sharing amongst colleagues!
- Actual learning/prediction task: this is where domain expertise from NBC, GDC, maybe even CBT colleagues, could come in handy! Thinking about this step right now is helpful for keeping us motivated, but we should probably focus on everything up till step 5.

### Allocation of work

We can always change later :). Be nimble!

- Automated homology modelling: Kannan, Mei
- Automated graph construction: Eric, Kannan
- Automated graph patch generation: Kannan, Eric
- GNN package: Eric, Mei
- Pretext task: Define later
- Actual learning task: Define later

### Artifacts to produce

Bitbucket and GitHub for code

- Protein Interaction Network: https://github.com/ericmjl/protein-interaction-network (source code related to converting PDB files to graphs)
- Protein Patch GNN: https://bitbucket.prd.nibr.novartis.net/projects/SDA/repos/sda-patch-gnn/browse (source code related to the graph neural network package).
- Jenkins + Docker for reproducible testing (minimum) and analyses (aspirational).

Project repository (can be the protein patch GNN at first) that houses all things related to the project.
