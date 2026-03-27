"""DFS sampling adapters."""

from clrs._src import dfs_sampling

sample_argmax = dfs_sampling.sample_argmax
sample_argmax_listofdict = dfs_sampling.sample_argmax_listofdict
sample_argmax_listofdatapoint = dfs_sampling.sample_argmax_listofdatapoint
sample_random_list = dfs_sampling.sample_random_list
sample_upwards = dfs_sampling.sample_upwards
sample_altUpwards = dfs_sampling.sample_altUpwards

__all__ = (
    "sample_argmax",
    "sample_argmax_listofdict",
    "sample_argmax_listofdatapoint",
    "sample_random_list",
    "sample_upwards",
    "sample_altUpwards",
)
