# Accurate, fast and automatic facial expression recognition

This repository is devoted to the development of the facial emotion recognition (FER) system under resource constraint as a final bachelor project at the TU/e. Realized by Blazej Manczak. Supervisors: Dr. Laura Astola and Dr. Vlado Menkovski. Please see the Report.pdf for a comprehensive overview of the methods used in this repository. A couple of things about this repo:

- I am using two datasets: **AffectNet** and **RAF-DB**
- The repo contains a preprocessing pipeline responsible for CLAHE illumination normalization, facial landmark detection (using MTCNN), face alignment, preprocessing as done for VGG-Face, or SE-ResNet-50 models, and face extraction. The scrirpt can be found in *BEP/ModelInference/withMtcnn*.
- The method described in the Report achieves close to the state-of-the-art results on the corresponding validation sets: **87.8%** on the RAF-DB dataset and **64.06%** on AffectNet.
- The method is capable of real-time processing: 4.5-5 FPS on a 2016 baseline MacBookPro 13" (CPU only).
- The model(s) can be successfully quantized with minimal accuracy loss, giving an opportunity of working in resource-constrained environments (2.5-3 FPS on Google Dev Board).
- Grad-Cam algorithm can be employed to create a coarse localization map highlighting important aspects of the image.

