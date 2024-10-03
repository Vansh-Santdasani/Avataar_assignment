

---

# Object Segmentation with SAM and GroundingDINO

Welcome to the **Object Segmentation** repository! This project leverages state-of-the-art models, **Segment Anything Model (SAM)** and **GroundingDINO**, to perform high-precision object segmentation using natural language prompts. Whether you're segmenting complex scenes or specific objects, this integration ensures both flexibility and robustness.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Model Downloads](#model-downloads)
- [Usage](#usage)
  - [Example Command](#example-command)
- [Output Examples](#output-examples)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Overview
This repository integrates **Segment Anything Model (SAM)** and **GroundingDINO** for precise segmentation of objects in images using text prompts. It uses GroundingDINO to identify bounding boxes based on textual descriptions and SAM to generate fine-grained segmentation masks, resulting in an accurate and intuitive output.

## Features
- **Text-Prompted Object Detection**: Identify and segment objects based on natural language descriptions.
- **Multi-Model Integration**: Seamlessly combines GroundingDINO for object detection and SAM for segmentation.
- **Output Flexibility**: Choose between a binary mask or a final image with a semi-transparent red mask overlay.
- **GPU Support**: Automatically utilizes GPU for faster inference if available.

## Setup

### Prerequisites
- Python 3.7 or higher
- CUDA (optional, for GPU acceleration)
- [PyTorch](https://pytorch.org/get-started/locally/) compatible with your system and CUDA version

### Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/object-segmentation.git
   cd object-segmentation
   ```

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the GroundingDINO library**:
   GroundingDINO needs to be cloned and installed separately. Run the following commands:
   ```bash
   git clone https://github.com/IDEA-Research/GroundingDINO.git
   cd GroundingDINO
   pip install -e .
   cd ..
   ```

4. **Install the Segment Anything library**:
   To install the `segment_anything` library, run:
   ```bash
   pip install git+https://github.com/facebookresearch/segment-anything.git
   ```

### Model Downloads
You need to download the pre-trained model weights for both SAM and GroundingDINO before running the script.

#### Segment Anything Model (SAM)
1. Go to the [SAM GitHub Repository](https://github.com/facebookresearch/segment-anything).
2. Download the `sam_vit_h_4b8939.pth` checkpoint:
   - Direct download link: [SAM Checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
3. Place the downloaded checkpoint file in the root directory of this project:
   ```
   /path/to/your/repo/sam_vit_h_4b8939.pth
   ```

#### GroundingDINO
1. Download the `GroundingDINO_SwinT_OGC.py` configuration file and the `groundingdino_swint_ogc.pth` checkpoint from the official repository:
   - **Config file**: [GroundingDINO SwinT Config](https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/config/GroundingDINO_SwinT_OGC.py)
   - **Checkpoint file**: [GroundingDINO SwinT Checkpoint](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0/groundingdino_swint_ogc.pth)
   
2. Place these files in the project directory structure:
   ```
   /path/to/your/repo/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
   /path/to/your/repo/groundingdino_swint_ogc.pth
   ```

### Directory Structure
After downloading the models, your directory structure should look like this:

```
object-segmentation/
├── GroundingDINO/
│   └── groundingdino/
│       └── config/
│           └── GroundingDINO_SwinT_OGC.py
├── sam_vit_h_4b8939.pth
├── groundingdino_swint_ogc.pth
├── run.py
├── examples/
│   ├── input_image.jpg
│   ├── output_image.png
├── README.md
└── requirements.txt
```

## Usage
You can run the script directly from the command line with the following format:

```bash
python run.py --image <path_to_image> --class <object_class> --output <output_image_path> [--mask-only]
```

### Parameters:
- **`--image`**: Path to the input image.
- **`--class`**: Class name of the object to segment (e.g., `person`, `dog`).
- **`--output`**: Path to save the output image.
- **`--mask-only`**: Optional flag to save only the binary mask instead of the overlay image.

### Example Command
```bash
python run.py --image examples/input_image.jpg --class "cat" --output examples/output_image.png
```

### Output
- **Original Image**: Displays the original input image.
- **Segmentation Mask**: Shows the binary mask for the detected object.
- **Final Image**: Saves the image with the red mask overlay.

## Output Examples
![Chair](https://github.com/user-attachments/assets/54ac86b0-89b3-4e01-8a8d-c74ea334b29f)
![mask](https://github.com/user-attachments/assets/8db1c77c-af24-4a4a-8837-d05e132c0379)
![generated](https://github.com/user-attachments/assets/13a6cd86-b48c-4cd3-8ff4-66211d85ecf4)


## Troubleshooting
1. **CUDA Error**: Ensure that CUDA is properly installed if you have a compatible GPU. If not, the script will automatically fall back to CPU.
2. **Model Files Not Found**: Make sure that the SAM and GroundingDINO model files are correctly downloaded and placed in the specified paths.

## Acknowledgements
This project leverages open-source models and code from:
- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)

