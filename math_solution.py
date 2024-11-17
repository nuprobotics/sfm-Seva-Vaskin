import numpy as np

def triangulation(
        camera_matrix: np.ndarray,
        camera_position1: np.ndarray,
        camera_rotation1: np.ndarray,
        camera_position2: np.ndarray,
        camera_rotation2: np.ndarray,
        image_points1: np.ndarray,
        image_points2: np.ndarray
):
    """
    :param camera_matrix: Camera intrinsic matrix, np.ndarray 3x3
    :param camera_position1: First camera position in world coordinate system, np.ndarray 3x1
    :param camera_rotation1: First camera rotation matrix in world coordinate system, np.ndarray 3x3
    :param camera_position2: Second camera position in world coordinate system, np.ndarray 3x1
    :param camera_rotation2: Second camera rotation matrix in world coordinate system, np.ndarray 3x3
    :param image_points1: Points in the first image, np.ndarray Nx2
    :param image_points2: Points in the second image, np.ndarray Nx2
    :return: Triangulated 3D points, np.ndarray Nx3
    """

    # Transpose rotation matrices to get rotations from world to camera coordinates
    R1 = camera_rotation1.T
    R2 = camera_rotation2.T

    # Compute translation vectors
    t1 = -R1 @ camera_position1.reshape(3, 1)
    t2 = -R2 @ camera_position2.reshape(3, 1)

    # Form extrinsic matrices
    E1 = np.hstack((R1, t1))
    E2 = np.hstack((R2, t2))

    # Compute projection matrices
    P1 = camera_matrix @ E1
    P2 = camera_matrix @ E2

    num_points = image_points1.shape[0]
    points_3d = np.zeros((num_points, 3))

    for i in range(num_points):
        x1, y1 = image_points1[i]
        x2, y2 = image_points2[i]

        # Construct matrix A for each point
        A = np.array([
            x1 * P1[2, :] - P1[0, :],
            y1 * P1[2, :] - P1[1, :],
            x2 * P2[2, :] - P2[0, :],
            y2 * P2[2, :] - P2[1, :]
        ])

        # Solve for X using SVD
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X /= X[-1]  # Convert to non-homogeneous coordinates

        points_3d[i] = X[:3]

    return points_3d
