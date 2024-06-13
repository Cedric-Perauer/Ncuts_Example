import open3d as o3d
import os
from normalized_cut import ncuts_chunk

config_spatial = {
    "alpha": 1.0,
    "T": 0.075, #ncuts threshold -> higher threshold = more recursive calls -> more cuts
}
input_folder = 'pcds_store/'
out_folder = 'out_pdcs/'
if os.path.exists(out_folder) == False :
    os.makedirs(out_folder)
    
for i in range(2):
    chunk_non_ground = o3d.io.read_point_cloud(f'{input_folder}non_ground{i}.pcd')
    chunk_ground = o3d.io.read_point_cloud(f'{input_folder}ground{i}.pcd').paint_uniform_color([0,0,0])
    chunk_major = o3d.io.read_point_cloud(f'{input_folder}chunk_major{i}.pcd')
    ncuts_out = ncuts_chunk(chunk_non_ground,chunk_ground,chunk_major,ncuts_threshold=config_spatial['T'],alpha=config_spatial['alpha'])
    o3d.io.write_point_cloud(f'{out_folder}/chunk_{i}.pcd',ncuts_out)
    try :
        o3d.visualization.draw_geometries([ncuts_out])
    except : 
        print("No display available, only storing the output")