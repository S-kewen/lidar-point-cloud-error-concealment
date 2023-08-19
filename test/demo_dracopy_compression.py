import numpy as np
import DracoPy
import time
import os
import open3d as o3d
from o3d_util import O3dUtil

def bin2ply(xyzr, output_path):
    xyz0 = xyzr[:, :3]
    points_i = xyzr[:, 3]
    point_list = o3d.geometry.PointCloud()
    point_list.points = o3d.utility.Vector3dVector(xyz0)
    point_list.colors = o3d.utility.Vector3dVector(O3dUtil.get_point_color(points_i))
    o3d.io.write_point_cloud(output_path, point_list)
    
def main():
    bin_file = "/mnt/data2/skewen/kittiGenerator/output/20230228_False_1_1_0_0_1000/object/training/velodyne/000000.bin"
    drc_file = "test.drc"
    
    points_xyzi = np.fromfile(bin_file, dtype=np.float32, count=-1).reshape([-1, 4])
    
    colors = np.concatenate((points_xyzi[:, 3].reshape(-1, 1), np.zeros((points_xyzi.shape[0], 2))), axis=1)
    colors = (colors * 255).astype(np.uint8)
    binary = DracoPy.encode(points_xyzi[:, :3], colors = colors, preserve_order = True)
    
    buffer_bin = np.frombuffer(binary, dtype=np.uint8)
    buffer_bin.tofile(drc_file)
    

    # binary = DracoPy.encode(
    # mesh.points, faces=mesh.faces,
    # quantization_bits=14, compression_level=1,
    # quantization_range=-1, quantization_origin=None,
    # create_metadata=False, preserve_order=False,
    # colors=mesh.colors
    # )

    compressed_drc = DracoPy.decode(np.fromfile(drc_file, dtype=np.uint8).tobytes())
    compressed_points_xyzi = np.concatenate((compressed_drc.points, compressed_drc.colors[:, 0].reshape(-1, 1) / 255), axis=1)
    
    print("lossless compression: {}".format(np.allclose(points_xyzi, compressed_points_xyzi)))
    compressed_points_xyzi.astype(np.float32).tofile("000000.bin")
    bin2ply(compressed_points_xyzi, "test.ply")
    print("points_xyzi shape: {}".format(points_xyzi.shape))
    print("compressed_points_xyzi shape: {}".format(compressed_points_xyzi.shape))
    print("Compression Ratio: {}".format(1 - (os.stat(drc_file).st_size / os.stat(bin_file).st_size)))
    print(points_xyzi - compressed_points_xyzi)
    
if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Total Running Time: {}".format(time.time()-start_time))