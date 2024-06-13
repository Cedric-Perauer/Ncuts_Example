## Minimum Ncuts Example 

```sh
pip install -r requirements.txt 
```

## Files 
- ```run_ncuts.ipynb & run_ncuts.py``` : main files to run and visualize some chunks, both files do the same
- ```normalized_cuts.py``` : contains main code for running ncuts
- ```point_cloud_utils.py``` : contains some helper functions for visualization and reprojecting points from downsampled pcd to original pointcloud

## Params 

This code is only using spatial distances, so the only relevant ncuts parameter is the ```ncuts_threshold T```. 
Increasing the threshold inside ```config_spatial``` dict will lead to more recursive splits (finer grained/more instances). 
