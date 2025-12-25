import matplotlib.pyplot as plt
import cv2
import numpy as np
from algorithms_library import detect_keypoints, draw_keypoints, read_images, apply_ratio_test

def q1(img1, img2):
    """
    Detects and displays keypoints for a stereo pair of images.

    Parameters:
        img1 (numpy.ndarray): The first image.
        img2 (numpy.ndarray): The second image.

    Returns:
        tuple: A tuple containing keypoints and descriptors for both images.
    """
    keypoints1, descriptors1 = detect_keypoints(img1, method='ORB', num_keypoints=500)
    keypoints2, descriptors2 = detect_keypoints(img2, method='ORB', num_keypoints=500)

    img1_with_keypoints = draw_keypoints(img1, keypoints1)
    img2_with_keypoints = draw_keypoints(img2, keypoints2)

    # Display images with keypoints
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img1_with_keypoints, cmap='gray')
    plt.title('Keypoints in Image 1')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img2_with_keypoints, cmap='gray')
    plt.title('Keypoints in Image 2')
    plt.axis('off')
    plt.show()
    return keypoints1, keypoints2, descriptors1, descriptors2


def q2(descriptors1, descriptors2):
    """
    Prints the first descriptor from each set of descriptors.

    Parameters:
        descriptors1 (numpy.ndarray): Descriptors for the first image.
        descriptors2 (numpy.ndarray): Descriptors for the second image.
    """
    print(descriptors1[0])
    print(descriptors2[0])


def q3(img1, img2, keypoints1, keypoints2, descriptors1, descriptors2):
    """
    Finds and displays matches between keypoints in two images.

    Parameters:
        img1 (numpy.ndarray): The first image.
        img2 (numpy.ndarray): The second image.
        keypoints1 (list): Keypoints detected in the first image.
        keypoints2 (list): Keypoints detected in the second image.
        descriptors1 (numpy.ndarray): Descriptors for the keypoints in the first image.
        descriptors2 (numpy.ndarray): Descriptors for the keypoints in the second image.

    Returns:
        list: List of matches between keypoints in the two images.
    """
    bf = cv2.BFMatcher()
    matches = bf.match(descriptors1, descriptors2)

    # Shuffle matches and select 20 random matches
    matches = list(matches)
    np.random.shuffle(matches)
    matches = matches[:20]

    # Draw matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display matches
    plt.figure(figsize=(12, 6))
    plt.imshow(img_matches)
    plt.title('20 Random Matches Between Key Points')
    plt.axis('off')
    plt.show()
    return matches


def q4(img1, img2, keypoints1, keypoints2, descriptors1, descriptors2):
    """
    Finds and displays matches between keypoints in two images after applying ratio test.

    Parameters:
        img1 (numpy.ndarray): The first image.
        img2 (numpy.ndarray): The second image.
        keypoints1 (list): Keypoints detected in the first image.
        keypoints2 (list): Keypoints detected in the second image.
        descriptors1 (numpy.ndarray): Descriptors for the keypoints in the first image.
        descriptors2 (numpy.ndarray): Descriptors for the keypoints in the second image.
    """
    bf = cv2.BFMatcher()
    good_matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = apply_ratio_test(good_matches)
    # Display 20 random matches
    np.random.shuffle(good_matches)
    good_matches = good_matches[:20]

    # Draw matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display matches
    plt.figure(figsize=(12, 6))
    plt.imshow(img_matches)
    plt.title('20 Matches After Ratio Test')
    plt.axis('off')
    plt.show()


def main():
    """
    Main function to execute the code.
    """
    img1, img2 = read_images(0)
    keypoints1, keypoints2, descriptors1, descriptors2 = q1(img1, img2)
    q2(descriptors1, descriptors2)
    matches = q3(img1, img2, keypoints1, keypoints2, descriptors1, descriptors2)
    q4(img1, img2, keypoints1, keypoints2, descriptors1, descriptors2)


if __name__ == "__main__":
    main()
