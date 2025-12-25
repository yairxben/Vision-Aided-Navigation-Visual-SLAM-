# Vision Aided Navigation (Visual SLAM)

## Project Description
This project implements a complete Visual Simultaneous Localization and Mapping (vSLAM) system designed to estimate the trajectory of a vehicle using stereo camera data. The system evolves from basic feature tracking into a robust navigation engine using factor graph optimization.

Key features include:
* **Feature Tracking:** Persistent tracking of 3D landmarks across multiple stereo frames using outlier rejection and PnP (Perspective-n-Point) algorithms.
* **Visual Odometry:** Initial trajectory estimation using frame-to-frame consensus matching.
* **Bundle Adjustment:** Local optimization of camera poses and 3D landmarks using a sliding window approach to refine the trajectory.
* **Pose Graph Optimization:** Global trajectory optimization that reduces drift by converting keyframes into a pose graph.
* **Loop Closure:** Detection of previously visited locations to add constraints to the global graph, significantly correcting accumulated drift.

## Motivation
Accurate self-localization is a critical challenge for autonomous vehicles, especially in "GPS-denied" environments (such as tunnels, urban canyons, or indoor spaces).

This project demonstrates how visual data alone can be used to navigate complex environments. By implementing advanced optimization techniques like Bundle Adjustment and Loop Closure, the system mitigates the "drift" inherent in dead-reckoning systems, ensuring that the estimated path remains consistent with the real world over long distances.

## Technologies & Libraries
* **Python:** Core logic and implementation.
* **GTSAM (Georgia Tech Smoothing and Mapping):** Used for solving complex factor graph optimizations (Bundle Adjustment, Pose Graph, and Loop Closure).
* **OpenCV:** Utilized for image processing tasks, including feature detection, matching, and triangulation.
* **NumPy:** Efficient numerical computation and matrix operations.
* **Matplotlib:** Visualization of the estimated 3D trajectory, camera poses, and tracking statistics.
