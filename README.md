# Vision Aided Navigation (Visual SLAM)

## Project Description
This project implements a complete Visual Simultaneous Localization and Mapping (vSLAM) system designed to estimate the trajectory of a vehicle using stereo camera data. The system evolves from basic feature tracking into a robust navigation engine using factor graph optimization.

## Motivation
Accurate self-localization is a critical challenge for autonomous vehicles, especially in "GPS-denied" environments (such as tunnels, urban canyons, or indoor spaces).

This project demonstrates how visual data alone can be used to navigate complex environments. By implementing advanced optimization techniques like Bundle Adjustment and Loop Closure, the system mitigates the "drift" inherent in dead-reckoning systems, ensuring that the estimated path remains consistent with the real world over long distances.


## Results Visualization

![Estimated Trajectory vs. Ground Truth](https://github.com/yairxben/Vision-Aided-Navigation-Visual-SLAM-/blob/main/Plots/Estimated%20Trajectory%20vs.%20Ground%20Truth.png?raw=true)

The figure above illustrates the step-by-step improvement of the vehicle's estimated trajectory compared to the actual path (Ground Truth):

* **1. Initial Estimation (Pink):** The raw trajectory calculated from frame-to-frame tracking. It suffers from significant drift over time as small errors accumulate.
* **2. Bundle Adjustment (Orange):** Represents the trajectory after applying local optimization. While smoother and locally consistent, it still deviates from the true path over long distances.
* **3. Loop Closure (Purple):** The final optimized trajectory. By detecting when the vehicle returns to a previous location and adding global constraints to the pose graph, the system eliminates the accumulated drift.
* **4. Ground Truth (Cyan):** The actual path taken by the vehicle. Note how the **Loop Closure** result overlaps almost perfectly with this reference.

Key features include:
* **Feature Tracking:** Persistent tracking of 3D landmarks across multiple stereo frames using outlier rejection and PnP (Perspective-n-Point) algorithms.
* **Visual Odometry:** Initial trajectory estimation using frame-to-frame consensus matching.
* **Bundle Adjustment:** Local optimization of camera poses and 3D landmarks using a sliding window approach to refine the trajectory.
* **Pose Graph Optimization:** Global trajectory optimization that reduces drift by converting keyframes into a pose graph.
* **Loop Closure:** Detection of previously visited locations to add constraints to the global graph, significantly correcting accumulated drift.

## Technologies & Libraries
* **Python:** Core logic and implementation.
* **GTSAM (Georgia Tech Smoothing and Mapping):** Used for solving complex factor graph optimizations (Bundle Adjustment, Pose Graph, and Loop Closure).
* **OpenCV:** Utilized for image processing tasks, including feature detection, matching, and triangulation.
* **NumPy:** Efficient numerical computation and matrix operations.
* **Matplotlib:** Visualization of the estimated 3D trajectory, camera poses, and tracking statistics.
