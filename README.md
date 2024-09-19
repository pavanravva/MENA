# MENA: Multimodal Epistemic Network Analysis for Analyzing Caregiver Attitudes in Augmented Reality

## Overview

MENA (Multimodal Epistemic Network Analysis) focuses on analyzing caregiver attitudes in augmented reality environments by utilizing a multimodal approach. The project aims to classify emotional states of caregivers based on their audio, gestures, and verbal communication during interactions captured in video segments.

<div align="center">
  <img src="model.jpg" alt="Flowchart of Data Processing" width="80%">
  <p><strong>Figure 1:</strong> ESP Model Framework: The architecture of the proposed Multimodal Emotional State Classifier consists of four key components: Audio Extraction, Pose Estimation, Text Features integrated with a Knowledge Graph, and a Fusion Network with a classification head. This framework assesses whether participants exhibit positive emotions—supportive and uplifting behaviors—during video segments.</p>
</div>

## Process

1. **Knowledge Graph Representation**: 
   - The text data is converted into a graph-like structure for knowledge graph representation. The text is sent to the ConceptNet API, which extracts information about related words and their relationships. 
   - This process utilizes the Knowledge Graph code, and the text data is also used to fine-tune the RoBERTa model for improved understanding.

2. **Data Extraction**: 
   - Audio data and 3D skeleton data are extracted and integrated with the text-derived features.

3. **Model Training**: 
   - All three modalities—text, audio, and 3D human pose data—are combined to train the model for emotional prediction.
