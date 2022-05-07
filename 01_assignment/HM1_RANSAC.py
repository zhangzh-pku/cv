import numpy as np
from utils import draw_save_plane_with_points

if __name__ == "__main__":

    # load data, total 130 points inlcuding 100 inliers and 30 outliers
    noise_points = np.loadtxt("HM1_ransac_points.txt")
    size = len(noise_points[..., 0])
    #RANSAC
    # we recommend you to formulate the palnace function as:  A*x+B*y+C*z+D=0

    sample_time = 11  #more than 99.9% probability at least one hypothesis does not contain any outliers
    distance_threshold = 0.05

    # sample points group
    index = np.random.choice(size, sample_time * 3)
    sample = noise_points[index].reshape((sample_time, 3, 3))
    #x1=sample[-1,0,0] y1=sample[-1,0,1] z1=sample[-1,0,2]
    # estimate the plane with sampled points group
    # a,b,c,d
    ans = np.zeros((sample_time, 4))
    #a=(y2-y1)*(z3-z1)-(y3-y1)*(z2-z1)
    ans[...,
        0] = (sample[..., 1, 1] -
              sample[..., 0, 1]) * (sample[..., 2, 2] - sample[..., 0, 2]) - (
                  sample[..., 2, 1] - sample[..., 0, 1]) * (sample[..., 1, 2] -
                                                            sample[..., 0, 2])
    #b=(z2-z1)*(x3-x1)-(z3-z1)*(x2-x1)
    ans[...,
        1] = (sample[..., 1, 2] -
              sample[..., 0, 2]) * (sample[..., 2, 0] - sample[..., 0, 0]) - (
                  sample[..., 2, 2] - sample[..., 0, 2]) * (sample[..., 1, 0] -
                                                            sample[..., 0, 0])
    #c=(x2-x1)*(y3-y1)-(x3-x1)*(y2-y1)
    ans[...,
        2] = (sample[..., 1, 0] -
              sample[..., 0, 0]) * (sample[..., 2, 1] - sample[..., 0, 1]) - (
                  sample[..., 2, 0] - sample[..., 0, 0]) * (sample[..., 1, 1] -
                                                            sample[..., 0, 1])
    #d=-a*x1-b*y1-c*z1
    ans[..., 3] = -(ans[..., 0] * sample[..., 0, 0]) - (
        ans[..., 1] * sample[..., 0, 1]) - (ans[..., 2] * sample[..., 0, 2])
    #evaluate inliers (with point-to-plance distance < distance_threshold)
    dis = ans[..., 0:3] @ (noise_points.T)
    dis = dis / np.tile(
        (np.sqrt(ans[..., 0]**2 + ans[..., 1]**2 + ans[..., 3]**2)).reshape(
            11, 1), size)
    threshold = np.sum(dis < distance_threshold, axis=1)
    is_max = np.argmax(threshold)
    # minimize the sum of squared perpendicular distances of all inliers with least-squared method
    pf = ans[is_max]
    print(pf)
    # draw the estimated plane with points and save the results
    # check the utils.py for more details
    # pf: [A,B,C,D] contains the parameters of palnace function  A*x+B*y+C*z+D=0

    draw_save_plane_with_points(pf, noise_points, "result/HM1_RANSAC_fig.png")
    np.savetxt("result/HM1_RANSAC_plane.txt", pf)
