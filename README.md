
## The data set


In this project, we have used the [Goodreads dataset](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home) collected by 
> Mengting Wan, Julian McAuley, "Item Recommendation on Monotonic Behavior Chains", RecSys 2018.


## Basic recommender system


Our recommendation model uses Spark's alternating least squares (ALS) method to learn latent factor representations for users and items.


### Data splitting and subsampling


  - Selected 60% of users (and all of their interactions) to form the *training set*.
  - Selected 20% of users to form the *validation set*.  For each validation user, use half of their interactions for training, and the other half should be held out for validation.  (Remember: you can't predict items for a user with no history at all!)
  - Remaining users: same process as for validation.


## Extensions


   - *Comparison to single-machine implementations*: compared Spark's parallel ALS model to a single-machine implementation, e.g. [lightfm](https://github.com/lyst/lightfm).  Our comparison measures both effeciency (model fitting time as a function of data set size) and resulting accuracy.
 

