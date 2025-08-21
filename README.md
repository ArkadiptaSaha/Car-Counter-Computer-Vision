# Car Counter using Computer Vision

## Objectives
The primary objective of this project was to design and implement a computer vision–based vehicle counting system that could accurately detect and track vehicles in real time. The goal was to overcome common challenges such as occlusion, overlapping vehicles, and repeated counting errors, thereby contributing to intelligent traffic management, congestion monitoring, and smart city development.

## Process
The system was developed using the **YOLOv8** pre-trained deep learning model for object detection, which enabled high-accuracy real-time identification of vehicles within a defined region of the road.  
A **virtual line** was placed as a counting threshold, with a tolerance of ±20 pixels, ensuring that vehicles were only counted once when crossing this line.

To address issues of multiple detections and repeated counts, the **SORT (Simple Online and Realtime Tracking)** algorithm was integrated. SORT assigns unique IDs to detected vehicles and tracks their trajectories across frames, preventing duplicate counts even in challenging scenarios with overlapping or closely moving vehicles.

The combination of YOLOv8 and SORT ensured a balance of speed, robustness, and reliability. The pipeline was implemented and tested on traffic video streams, simulating real-world conditions for scalability and deployment.

## Outcomes
The developed system successfully demonstrated **accurate vehicle counting**, even under difficult conditions such as occlusion and overlapping cars. By leveraging YOLOv8 for real-time detection and SORT for unique ID-based tracking, the system minimized repeated counting errors and improved consistency.  

The approach provided a **scalable, cost-effective, and reliable solution** for traffic monitoring, supporting smarter infrastructure planning and safer road networks.
