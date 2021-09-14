20210809

After talking to Cihan, I realized that softmax on non-zero element was the way to go for GAT instead of tanh. 

I have tried to implmented the softmax_on_non_zero on gat branch..but need to add the exponential to it before doing the np.sum(axis =1). However, this implmentation considers only nearby neighbors --> attention * adjcency_matrix to mask out the non-connected neighbors. B/c of that we have to apply normalization on non-zero elements using `jnp.nonzero`. However this function does not work with jit (our majaroty of the pipeline requires jit) so it could not scale.

Cihan offered me to do a pair-coding session on dgl pytorch package, dgl pytorch package has varities of GAT layers. Will try that.
