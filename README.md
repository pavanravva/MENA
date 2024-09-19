# MENA: Multimodal Epistemic Network Analysis for Analyzing Caregiver Attitudes in Augmented Reality



<div align="center"> <img src="model.jpg" alt="Flowchart of Data Processing" width="80%"> <p><strong>Figure 1:</strong> ESP Model Framework: The architecture of the proposed Multimodal Emotional State Classifier consists of four key components: Audio Extraction, Pose Estimation, Text Features integrated with a Knowledge Graph, and a Fusion Network with a classification head. This framework assesses whether participants exhibit positive emotions—supportive and uplifting behaviors—during video segments, based on their audio, gestures, and verbal communication.</p> </div>

# Process
Knowledge Graph Representation: The text data is converted into a graph-like structure for knowledge graph representation. The text is sent to the ConceptNet API, which extracts information about related words and their relationships. This process utilizes the Knowledge Graph code, and the text data is also used to fine-tune the RoBERTa model for better understanding.

Data Extraction: Audio data and 3D skeleton data are extracted and integrated with the text-derived features.

Model Training: All three modalities—text, audio, and 3D human pose data—are combined to train the model for emotional prediction.



