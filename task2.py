###############
##Design the function "calibrate" to  return 
# (1) intrinsic_params: should be a list with four elements: [f_x, f_y, o_x, o_y], where f_x and f_y is focal length, o_x and o_y is offset;
# (2) is_constant: should be bool data type. False if the intrinsic parameters differed from world coordinates. 
#                                            True if the intrinsic parameters are invariable.
#It is ok to add other functions if you need
###############
import numpy as np
from cv2 import imread, cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners

wc = [[40, 0, 40], [40, 0, 30], [40, 0, 20], [40, 0, 10], 
      [30, 0, 40], [30, 0, 30], [30, 0, 20], [30, 0, 10], 
      [20, 0, 40], [20, 0, 30], [20, 0, 20], [20, 0, 10], 
      [10, 0, 40], [10, 0, 30], [10, 0, 20], [10, 0, 10], 
      [0, 0, 40], [0, 0, 30], [0, 0, 20], [0, 0, 10], 
      [0, 10, 40], [0, 10, 30], [0, 10, 20], [0, 10, 10], 
      [0, 20, 40], [0, 20, 30], [0, 20, 20], [0, 20, 10], 
      [0, 30, 40], [0, 30, 30], [0, 30, 20], [0, 30, 10], 
      [0, 40, 40], [0, 40, 30], [0, 40, 20], [0, 40, 10]]
wc = np.array(wc, dtype=np.float32)

def calibrate(imgname, w_c = wc):
  '''
  Calibrate camera to get the intrinsic matrix.

  Input
  -------
  imgname[str] : path to image file
  w_c[ndarray] : array of 3-d co-ordinates


  Returns
  -------
  intrinsic_params[ndarray] : array of intrinsic parameters of a camera
  is_constant[bool] : True, the intrinsic parameters of the camera do not depend on how we take the corresponding point of world origin and is a property of the camera itself
  '''

  img = imread(imgname)
  retval, corners = findChessboardCorners(cvtColor(img, COLOR_BGR2GRAY), (4,9))

  criteria = (TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 50, 0.0001)
  prec_corners = cornerSubPix(cvtColor(img, COLOR_BGR2GRAY), corners, (7,7),(-1,-1), criteria)
  # prec_corners = corners

  n = len(prec_corners)

  prec_corners = prec_corners.reshape((n,-1))

  A = np.zeros((2*n, 12), dtype = np.float32)

  for i in range(n):
    X, Y, Z = wc[i]
    x, y = prec_corners[i]
    row1 = np.array([X, Y, Z, 1, 0, 0, 0, 0, -1*x*X, -1*x*Y, -1*x*Z, -1*x])
    row2 = np.array([0, 0, 0, 0, X, Y, Z, 1, -1*y*X, -1*y*Y, -1*y*Z, -1*y])
    A[2*i] = row1
    A[2*i+1] = row2

  U, S, V_T = np.linalg.svd(A)
  x = V_T[-1].reshape((3,4))

  x3 = x[-1][:3]
  k = np.sqrt(np.sum(x3*x3)) #np.linalg.norm(x3)

  m = (1/k)*x

  m1 = m[0][:3]
  m2 = m[1][:3]
  m3 = m[2][:3]
  m4 = m[:,3]

  o_x, o_y = np.matmul(m1.T, m3), np.matmul(m2.T, m3)
  f_x, f_y = np.sqrt(np.matmul(m1.T,m1)-o_x*o_x), np.sqrt(np.matmul(m2.T,m2)-o_y*o_y)

  # intrinsic_params = np.identity((3))
  # intrinsic_params[0][0] = f_x
  # intrinsic_params[0][2] = o_x
  # intrinsic_params[1][1] = f_y
  # intrinsic_params[1][2] = o_y

  intrinsic_params = np.array([f_x, f_y, o_x, o_y], dtype = np.float32)

  return intrinsic_params, True

if __name__ == "__main__":
    intrinsic_params, is_constant = calibrate('checkboard.png')
    print(intrinsic_params)
    print(is_constant)