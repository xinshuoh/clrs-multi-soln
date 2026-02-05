# june 10
# figure out the difference between upwards and altupwards sampling
# in correctness and runtime

from clrs._src import dfs_sampling

# ----------------------------------------------
# test extract_probMatrix
# ----------------------------------------------
# extract_probMatrix expects a list of datapoints, each objects whose .data attribute is a list of distributions
# So an object with .data being a list of list of lists.
class dummydatapoint:
    def __init__(self, data):
        self.data = data

DISTLIST = [
    [
        [0,0,0],
        [0,0,0],
        [0,0,0]
    ],
    [
        [1,1,1],
        [1,1,1],
        [1,1,1]
    ]
]

test = dummydatapoint(DISTLIST)
test2 = dummydatapoint(DISTLIST)

myinput = [test, test2]

big_distlist = dfs_sampling.extract_probMatrices(myinput)
assert(big_distlist[2] == DISTLIST[0])

# ----------------------------------------------
# test upwards
# ----------------------------------------------
# need a big-enough graph to expose runtime differences...



# ----------------------------------------------
# test alt_upwards
# ----------------------------------------------