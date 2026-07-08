# Vision-Based Modular Assembly Monitoring System

A modular computer vision framework for monitoring manual assembly processes using classical computer vision techniques, developed as part of my **Master of Science Thesis in Aerospace Engineering** at **TU Darmstadt, Germany**.

---

# Project Overview

This project focuses on the development of a scalable and modular software architecture for industrial assembly monitoring. The objective is to analyse manual assembly operations, detect worker activities, measure process performance, and provide a foundation for future AI-based task recognition.

The software has been designed using independent subsystems to improve maintainability, scalability, testing, and future integration of advanced computer vision and artificial intelligence algorithms.

The current implementation represents the first working prototype (Version 1), which is based on classical computer vision techniques. Future versions will extend the system using deep learning, synthetic data generation, Digital Twin integration, and industrial communication protocols.

---

# Research Objectives

The primary objectives of this research are:

- Develop a modular software architecture for industrial assembly monitoring.
- Detect worker motion using classical computer vision.
- Analyse assembly events and process performance.
- Design a scalable framework for future AI integration.
- Support future Human Activity Recognition (HAR).
- Enable future Digital Twin integration.
- Prepare the software for future OPC UA communication.
- Build a reusable vision pipeline for industrial applications.

---

# Current Version

## Version 1 – Classical Computer Vision Prototype

### Current Status

✅ Modular software architecture completed

✅ Camera abstraction layer implemented

✅ Logitech webcam integration

✅ Intel RealSense interface implemented

✅ ROI selection

✅ Frame validation

✅ Motion detection

✅ Motion score calculation

✅ Motion state estimation

✅ Event detection

✅ Analytics engine

✅ CSV logging

✅ Live visualization

✅ End-to-end software pipeline operational

---

# Software Architecture

The software follows a modular pipeline architecture where each subsystem performs a specific task independently. This design allows future algorithms (such as YOLO) to be integrated without redesigning the complete software.

```text
Configuration
        │
        ▼
Camera Acquisition
        │
        ▼
Frame Validation
        │
        ▼
ROI Processing
        │
        ▼
Grayscale Conversion
        │
        ▼
Gaussian Blur
        │
        ▼
Frame Difference
        │
        ▼
Thresholding
        │
        ▼
Morphological Filtering
        │
        ▼
Contour Detection
        │
        ▼
Motion Score Calculation
        │
        ▼
Motion State Estimation
        │
        ▼
Event Detection
        │
        ▼
Analytics
        │
        ▼
Storage
        │
        ▼
Visualization
```

---

# Project Structure

```text
Assembly_monitor/

├── acquisition/       Camera interfaces
├── analytics/         Performance metrics
├── calibration/       ROI calibration
├── config/            Configuration settings
├── decision/          Event detection logic
├── detection/         Motion detection modules
├── docs/              Documentation
├── images/            Images and figures
├── logs/              Generated CSV logs
├── processing/        Image processing pipeline
├── recordings/        Recorded videos
├── storage/           Data logging
├── validation/        Frame validation
├── visualization/     Live GUI

├── main.py            Main application
└── README.md          Project documentation
```

---

# Current Features

- Modular software architecture
- Camera abstraction layer
- Logitech webcam support
- Intel RealSense camera interface
- Region of Interest (ROI) selection
- Frame validation
- Motion detection
- Motion score estimation
- Motion state estimation
- Event detection
- Real-time analytics
- CSV session logging
- Live visualization
- Modular subsystem communication

---

# Current Limitations

The current prototype is based on classical computer vision techniques.

Although the system can detect motion, calculate motion scores, estimate motion states, and identify assembly events, it cannot yet understand the semantic meaning of worker actions.

For example, the software cannot distinguish between actions such as:

- Pick
- Place
- Insert
- Tighten
- Inspect

These capabilities will be introduced in the next prototype using deep learning models.

---

# Development Roadmap

## Version 1 (Completed)

- Classical Computer Vision Prototype
- Modular Software Architecture
- Motion Detection
- Event Detection
- Analytics
- Logging
- Live Visualization

---

## Version 2 (In Development)

- YOLO Object Detection
- Assembly Task Recognition
- Assembly Sub-step Tracking
- Ground Truth Generation
- Dataset Collection
- Image Annotation
- Training Pipeline
- AI Integration

---

## Version 3 (Future Work)

- Design of Experiments (DOE)
- Parameter Optimisation
- Performance Evaluation
- Robustness Testing
- Validation Metrics

---

## Version 4 (Future Work)

- Intel RealSense Depth Processing
- Human Pose Estimation
- Synthetic Dataset Generation
- Digital Twin Integration
- OPC UA Communication

---

# Technologies Used

- Python
- OpenCV
- NumPy
- Intel RealSense SDK
- Logitech Webcam
- Visual Studio Code
- Git
- GitHub

Future versions will additionally include:

- YOLO
- PyTorch
- Label Studio / CVAT
- Blender / Isaac Sim
- OPC UA

---

# Future Research Direction

The long-term objective of this research is to transform the current motion-based monitoring framework into an intelligent assembly monitoring system capable of:

- Recognising assembly tasks
- Tracking assembly sub-steps
- Measuring productivity
- Evaluating ergonomics
- Supporting Digital Twin applications
- Enabling Industry 5.0 manufacturing

---

# Author

**Mehul Patil**

M.Sc. Aerospace Engineering

Technical University of Darmstadt (TU Darmstadt)

Germany

---

# Project Status

**Current Status:** Active Development

Current Milestone:

**Version 1 Completed**

Next Milestone:

**Version 2 – AI-Based Assembly Task Recognition using YOLO**