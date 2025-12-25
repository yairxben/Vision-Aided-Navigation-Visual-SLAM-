import matplotlib.pyplot as plt
import cv2
import numpy as np

from algorithms_library import read_cameras, reject_matches, init_matches, cv_triangulation, triangulation_process


def main():
    deviations, img1_color, img2_color, inliers, keypoints1, keypoints2, outliers = q1()
    q2(deviations, img1_color, img2_color, inliers, keypoints1, keypoints2, outliers)
    q3(inliers, keypoints1, keypoints2)
    q4()

def q1():
    img1_color, img2_color, keypoints1, keypoints2, matches = init_matches(0)
    deviations, inliers, outliers = reject_matches(keypoints1, keypoints2, matches)
    print(len(inliers))
    plt.figure(figsize=(10, 5))
    plt.hist(deviations, bins=30, edgecolor='black')
    plt.title('Histogram of Deviations from Horizontal Line in Stereo Matches')
    plt.xlabel('Deviation (pixels)')
    plt.ylabel('Frequency')
    plt.show()
    return deviations, img1_color, img2_color, inliers, keypoints1, keypoints2, outliers


def q2(deviations, img1_color, img2_color, inliers, keypoints1, keypoints2, outliers):
    for match in inliers:
        pt1 = keypoints1[match.queryIdx].pt
        pt2 = keypoints2[match.trainIdx].pt
        cv2.circle(img1_color, (int(pt1[0]), int(pt1[1])), 2, (0, 165, 255), -1)
    cv2.circle(img2_color, (int(pt2[0]), int(pt2[1])), 2, (0, 165, 255), -1)
    for match in outliers:
        pt1 = keypoints1[match.queryIdx].pt
        pt2 = keypoints2[match.trainIdx].pt
        cv2.circle(img1_color, (int(pt1[0]), int(pt1[1])), 3, (255, 255, 0), -1)
        cv2.circle(img2_color, (int(pt2[0]), int(pt2[1])), 3, (255, 255, 0), -1)
    img_combined = np.vstack((img1_color, img2_color))
    plt.figure(figsize=(10, 20))
    plt.imshow(cv2.cvtColor(img_combined, cv2.COLOR_BGR2RGB))
    plt.title('Inliers (orange) and Outliers (cyan) Matches')
    plt.axis('off')
    plt.show()
    num_outliers = len(outliers)
    percentage_outliers = (num_outliers / len(deviations)) * 100
    print(f'Percentage of matches with deviations greater than 2 pixels: {percentage_outliers:.2f}%')
    print(f'Number of matches discarded: {num_outliers}')


def q3(inliers, keypoints1, keypoints2):
    k, P0, P1 = (
        read_cameras('C:/Users/avishay/PycharmProjects/SLAM_AVISHAY_YAIR/VAN_ex/dataset/sequences/00/calib.txt'))
    points_3D_custom, pts1, pts2 = triangulation_process(P0, P1, inliers, k, keypoints1, keypoints2)
    points_3D_cv = cv_triangulation(k @ P0, k @ P1, pts1, pts2)
    distances = np.linalg.norm(points_3D_custom - points_3D_cv, axis=1)
    median_distance = np.median(distances)
    print(f'Median distance between custom and OpenCV triangulated points: {median_distance:.10f} units')


def q4():
    for i in range(2):
        img1_color, img2_color, keypoints1, keypoints2, matches = init_matches(i)
        deviations, inliers, outliers = reject_matches(keypoints1, keypoints2, matches)
        k, P0, P1 = (
            read_cameras('C:/Users/avishay/PycharmProjects/SLAM_AVISHAY_YAIR/VAN_ex/dataset/sequences/00/calib.txt'))
        triangulation_process(P0, P1, inliers, k, keypoints1, keypoints2)

if __name__ == "__main__":
    main()
