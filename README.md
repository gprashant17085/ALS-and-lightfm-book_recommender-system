
## The data set

In this project, we have used the [Goodreads dataset](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home) collected by 
> Mengting Wan, Julian McAuley, "Item Recommendation on Monotonic Behavior Chains", RecSys 2018.


On Dumbo's HDFS, you will find the following files in `hdfs:/user/bm106/pub/goodreads`:

  - `goodreads_interactions.csv`
  - `user_id_map.csv`
  - `book_id_map.csv`

The first file contains tuples of user-book interactions.  For example, the first five linrd are
```
user_id,book_id,is_read,rating,is_reviewed
0,948,1,5,0
0,947,1,5,1
0,946,1,5,0
0,945,1,5,0
```

The other two files consist of mappings between the user and book numerical identifiers used in the interactions file, and their alphanumeric strings which are used in supplementary data (see below).
Overall there are 876K users, 2.4M books, and 223M interactions.

## Basic recommender system

Our recommendation model uses Spark's alternating least squares (ALS) method to learn latent factor representations for users and items.

### Data splitting and subsampling


  - Selected 60% of users (and all of their interactions) to form the *training set*.
  - Selected 20% of users to form the *validation set*.  For each validation user, use half of their interactions for training, and the other half should be held out for validation.  (Remember: you can't predict items for a user with no history at all!)
  - Remaining users: same process as for validation.



## Extensions



   - *Comparison to single-machine implementations*: compared Spark's parallel ALS model to a single-machine implementation, e.g. [lightfm](https://github.com/lyst/lightfm).  Our comparison measures both effeciency (model fitting time as a function of data set size) and resulting accuracy.
 

