import open3d as o3d
import numpy as np

def ply2rgbd(ply_file, depth_scale = 1.0, depth_trunc = 3.0, convert_rgb_to_intensity = False):
    """Convert a PLY file to Open3D RGBDImage.
    Parameters
    ----------
    ply_file : str
        Path to the PLY file.
    depth_scale : float
        Scale factor for depth.
    depth_trunc : float
        Truncation threshold for depth.
    convert_rgb_to_intensity : bool
        If set, the RGB values are averaged to compute
        intensity.
    Returns
    -------
    o3d.geometry.RGBDImage
        RGBDImage of the PLY file.
    """
    # Read the PLY file
    pcd = o3d.io.read_point_cloud(ply_file)
    # Convert to numpy array
    xyz = np.asarray(pcd.points).astype(np.float32)
    rgb = np.asarray(pcd.colors).astype(np.float32)
    # Get depth
    depth = xyz[:, 2].reshape(-1, 1)
    # Truncate depth
    depth[depth > depth_trunc] = 0
    # Scale depth
    depth = depth * depth_scale
    # Get intensity from RGB
    intensity = np.mean(rgb, axis=1).reshape(-1, 1)
    # Create rgbd image
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb),
        o3d.geometry.Image(depth),
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=convert_rgb_to_intensity
    )
    return rgbd

img = ply2rgbd("/mnt/data2/skewen/kittiGenerator/output/test_1_50_0_10/object/training/ply/000000.ply")
o3d.io.write_image("copy_of_lena_color.jpg", img)
