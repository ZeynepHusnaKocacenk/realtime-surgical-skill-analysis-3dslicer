# realtime-surgical-skill-analysis-3dslicer

# Real-Time Sequence Data Analysis in 3D Slicer Using Deep Neural Networks

A modular, end-to-end deep learning pipeline for real-time surgical 
skill classification using transformation data from motion tracking systems.

## Overview
This system integrates PyTorch-based BiLSTM and TCN models directly 
into 3D Slicer, enabling real-time inference on live tool motion data 
with sub-25ms latency. Built as part of a B.Sc. thesis at Carleton 
University (AI/ML Stream).

## Key Features
- Real-time inference pipeline achieving sub-25ms latency
- Modular architecture supporting plug-and-play model replacement
- Online (real-time) and offline (training) pipeline modes
- Integration with 3D Slicer via custom scripted modules
- Simulated dual-instance streaming using OpenIGTLink
- Data augmentation: window warping, slicing, jittering
- BiLSTM and TCN model support

## Technologies
Python, PyTorch, 3D Slicer, OpenIGTLink, NumPy, Qt, VTK, OptiTrack

## Demo
[YouTube Demo Video Link]

## Dataset
Ultrasound Needle Dataset (Xia et al., 2018)
