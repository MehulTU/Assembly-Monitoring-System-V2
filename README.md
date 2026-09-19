# AI-Supported Modular Assembly Monitoring System (Version 2)

> **Project family:** [V1 Classical CV](https://github.com/MehulTU/Assembly-Monitoring-System) · **V2 YOLO11 pipeline (this repo)** · [V2.1 upgrade](https://github.com/MehulTU/Assembly-Monitoring-System-V2.1) · Author: [@MehulTU](https://github.com/MehulTU)

An AI-supported modular computer vision framework for monitoring manual assembly processes using **YOLO-based object detection**, developed as part of my **Master of Science Thesis in Aerospace Engineering** at **Technical University of Darmstadt (TU Darmstadt), Germany**.

Version 2 extends the validated classical computer vision architecture developed in Version 1 by introducing deep learning for object recognition while preserving the modular software design. The framework is designed to support future research in intelligent manufacturing, productivity analysis, ergonomics assessment, and Digital Twin integration.

---

# Project Overview

The objective of this project is to develop a scalable and modular AI-based vision system capable of understanding manual assembly operations through object recognition, process-state estimation, and event analysis.

Unlike Version 1, which relied solely on motion analysis, Version 2 enables semantic understanding of the assembly process by identifying individual assembly components using a trained YOLO model.

The software architecture remains modular, allowing new AI models, sensors, and industrial communication interfaces to be integrated without redesigning the existing system.

---

# Research Objectives

The primary objectives of this research are:

- Develop a modular AI-supported assembly monitoring framework.
- Integrate YOLO object detection into the existing modular architecture.
- Recognize assembly components in real time.
- Estimate assembly process states.
- Generate process events from detected objects.
- Measure productivity and process performance.
- Build a reusable dataset generation and annotation pipeline.
- Develop a quantitative validation framework using ground truth.
- Prepare the software for future Digital Twin integration.
- Enable future Industry 5.0 manufacturing applications.

---

# Current Version

## Version 2 – AI-Based Assembly Monitoring Prototype

### Current Status

✅ Modular software architecture retained from Version 1

✅ YOLO11 object detection integrated

✅ Custom dataset generation pipeline

✅ Automated frame extraction

✅ Dataset quality assessment

✅ Near-duplicate detection

✅ Manual dataset review workflow

✅ Curated dataset generation

✅ YOLO dataset preparation

✅ Custom YOLO training pipeline

✅ Offline inference pipeline

✅ Temporal detection filtering

✅ Assembly state estimation

✅ Event logging

✅ Analytics generation

✅ Ground-truth generation tools

✅ Quantitative validation pipeline

✅ Validation error diagnosis

---

# System Architecture

Version 2 preserves the modular architecture developed in Version 1 while replacing classical motion-based perception with AI-based object recognition.

```text
Configuration
        │
        ▼
Acquisition
        │
        ▼
Calibration
        │
        ▼
YOLO Object Detection
        │
        ▼
Temporal Filtering
        │
        ▼
Assembly State Estimation
        │
        ▼
Decision Engine
        │
        ▼
Analytics
        │
        ▼
Storage
        │
        ▼
Validation
        │
        ▼
Visualization
```

---

# Project Structure

```text
Assembly_monitor/
│
├── acquisition/          Camera interfaces
├── analytics/            Process metrics
├── calibration/          Camera calibration
├── config/               Configuration management
├── decision/             Assembly state logic
├── processing/           Detection pipeline
├── storage/              Data storage
├── validation/           Quantitative validation
├── visualization/        Result visualization
├── docs/                 Documentation
├── images/               Figures
├── logs/                 Runtime logs
├── recordings/           Recorded videos
│
├── main.py               Main software pipeline
└── README.md

datasets/
experiments/
scripts/
weights/
models/
runs/
recordings/
```

---

# Current Features

- Modular software architecture
- YOLO11 object detection
- AI-supported assembly monitoring
- Custom dataset recording
- Frame extraction pipeline
- Dataset quality analysis
- Annotation workflow
- YOLO training pipeline
- Offline inference
- Temporal filtering
- Assembly state estimation
- Event detection
- Process analytics
- Quantitative validation
- Error diagnosis
- Performance evaluation

---

# Current Limitations

The current implementation recognizes trained assembly objects but is limited to the available dataset.

The system currently supports:

- Marker detection
- Power adapter detection

Future versions will extend the framework to:

- Complex industrial assembly components
- Multi-object assembly sequences
- Human pose estimation
- 3D object understanding
- Ergonomic assessment
- Real-time deployment

---

# Development Roadmap

## Version 1 (Completed)

- Classical Computer Vision
- Motion Detection
- Event Detection
- Analytics Framework
- Modular Architecture

---

## Version 2 (Current)

- YOLO Object Detection
- AI-based Assembly Monitoring
- Dataset Generation
- Annotation Pipeline
- Training Pipeline
- Offline Inference
- Validation Framework
- Error Diagnosis

---

## Version 3 (Future)

- Real Industrial Assembly Components
- Multi-class Detection
- Assembly Sequence Recognition
- Robustness Evaluation
- Design of Experiments (DOE)

---

## Version 4 (Future)

- Intel RealSense Integration
- Human Pose Estimation
- Synthetic Dataset Generation
- Digital Twin Integration
- OPC UA Communication
- ROS2 Integration
- Edge AI Deployment

---

# Technologies Used

## Core Technologies

- Python
- OpenCV
- YOLO11
- Ultralytics
- PyTorch
- NumPy

## Development Tools

- Visual Studio Code
- Git
- GitHub

## Dataset Tools

- CVAT
- Custom Annotation Pipeline

## Future Technologies

- Intel RealSense
- MediaPipe
- Blender
- Isaac Sim
- ROS2
- OPC UA
- Digital Twin

---

# Future Research Direction

The long-term objective is to evolve the framework into a complete AI-supported assembly monitoring platform capable of:

- Assembly task recognition
- Assembly sequence understanding
- Productivity analysis
- Ergonomic risk assessment
- Digital Twin synchronization
- Human-AI collaboration
- Industry 5.0 manufacturing

---

# Author

**Mehul Patil**

M.Sc. Aerospace Engineering

Technical University of Darmstadt (TU Darmstadt)

Germany

---

# Project Status

**Current Status:** Active Development

**Completed Milestone**

✅ Version 1 – Classical Computer Vision Framework

**Current Milestone**

🚧 Version 2 – AI-Supported Modular Assembly Monitoring using YOLO

**Next Milestone**

🎯 Version 3 – Real Industrial Assembly Monitoring using Aerospace Assembly Components
