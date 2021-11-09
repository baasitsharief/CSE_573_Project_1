###############
##Design the function "findRotMat" to  return 
# 1) rotMat1: a 2D numpy array which indicates the rotation matrix from xyz to XYZ 
# 2) rotMat2: a 2D numpy array which indicates the rotation matrix from XYZ to xyz 
#It is ok to add other functions if you need
###############

import numpy as np
import cv2

def Rx(theta):
    '''
    Rotation matrix for theta degrees rotation around X axis

    Input
    -------
    theta[float] : Rotation angle

    Returns
    -------
    R[ndarray] : Rotation Matrix
    '''
    R = np.identity((3))
    R[1][1] = np.cos(np.deg2rad(theta))
    R[2][1] = np.sin(np.deg2rad(theta))
    R[1][2] = -np.sin(np.deg2rad(theta))
    R[2][2] = np.cos(np.deg2rad(theta))
    return R

def Ry(theta):
    '''
    Rotation matrix for theta degrees rotation around Y axis

    Input
    -------
    theta[float] : Rotation angle

    Returns
    -------
    R[ndarray] : Rotation Matrix
    '''
    R = np.identity((3))
    R[0][0] = np.cos(np.deg2rad(theta))
    R[0][2] = np.sin(np.deg2rad(theta))
    R[2][0] = -np.sin(np.deg2rad(theta))
    R[2][2] = np.cos(np.deg2rad(theta))
    return R

def Rz(theta):
    '''
    Rotation matrix for theta degrees rotation around Z axis

    Input
    -------
    theta[float] : Rotation angle

    Returns
    -------
    R[ndarray] : Rotation Matrix
    '''
    R = np.identity((3))
    R[0][0] = np.cos(np.deg2rad(theta))
    R[0][1] = -np.sin(np.deg2rad(theta))
    R[1][0] = np.sin(np.deg2rad(theta))
    R[1][1] = np.cos(np.deg2rad(theta))
    return R

def findRotMat(alpha, beta, gamma):
    '''
    Finds rotation matrices for xyz to XYZ and XYZ to xyz co-ordinate systems respectively i.e. zx'z'' transformation and Zx'z' transformation.

    Input
    -------
    alpha[float] : angle of rotation around z axis for zx'z''
    beta[float] : angle of rotation around x' axis for zx'z''
    gamma[float] : angle of rotation around z'' axis for zx'z''

    Returns
    -------
    xyz2XYZ[ndarray] : Rotation Matrix to get XYZ from xyz
    XYZ2xyz[ndarray] : Rotation Matrix to get xyz from XYZ
    '''
    xyz2XYZ = np.matmul(np.matmul(Rz(gamma), Rx(beta)), Rz(alpha))
    XYZ2xyz = np.matmul(np.matmul(Rz(-alpha), Rx(-beta)), Rz(-gamma))
    return xyz2XYZ, XYZ2xyz

if __name__ == "__main__":
    alpha = 45
    beta = 30
    gamma = 60
    rotMat1, rotMat2 = findRotMat(alpha, beta, gamma)
    print(rotMat1)
    print(rotMat2)
