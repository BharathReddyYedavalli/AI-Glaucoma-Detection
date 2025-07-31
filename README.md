# AI Glaucoma Detection

A deep learning project for automated glaucoma detection in retinal images using state-of-the-art computer vision models. This project implements multiple CNN architectures and provides an interactive web interface for real-time glaucoma screening with explainable AI visualizations.

## Overview

Glaucoma is a leading cause of irreversible blindness worldwide. Early detection is crucial for preventing vision loss. This project leverages deep learning to automatically analyze retinal images and detect signs of glaucoma, providing:

- **High-accuracy detection** using EfficientNetV2-S architecture
- **Explainable AI** with Grad-CAM heatmap visualizations
- **Interactive web interface** for easy image analysis
- **Multiple model comparisons** (EfficientNetV2, ResNet50, MobileNetV3)

## Model Performance

| Model | Architecture | Accuracy | Key Features |
|-------|-------------|----------|--------------|
| **EfficientNetV2-S**  | State-of-the-art | Best Performance | Optimal speed/accuracy balance |
| ResNet50 | Deep Residual | High Accuracy | Proven architecture |
| MobileNetV3-Large | Lightweight | Fast Inference | Mobile-optimized |

## Quick Start

### Prerequisites
- Python 3.10+
- Windows/Linux/MacOS
- 4GB+ RAM recommended

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/BharathReddyYedavalli/AI-Glaucoma-Detection.git
cd AI-Glaucoma-Detection
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
python main.py
```

4. **Access the interface:**
   - Local: `http://localhost:7865`
   - Public URL will be displayed in terminal

## Web Interface Features

- **Drag & Drop Upload**: Easy image upload
- **Real-time Analysis**: Instant glaucoma detection
- **Grad-CAM Visualization**: See what the AI focuses on
- **Confidence Scores**: Prediction confidence levels
- **Medical Explanations**: Detailed analysis reports

## Model Architecture

### EfficientNetV2-S (Primary Model)
```python
Input: 288x288 RGB retinal images
├── EfficientNetV2-S backbone (pre-trained)
├── Global Average Pooling
├── Dropout (0.5)
├── Linear (1280 → 512)
├── ReLU activation
├── Dropout (0.3)
└── Linear (512 → 2) [Normal, Glaucoma]
```

## Technical Details

### Data Processing
- **Input Resolution**: 288×288 pixels
- **Normalization**: ImageNet statistics
- **Augmentation**: Training-time augmentations applied
- **Format Support**: JPG, PNG, TIFF

### Explainable AI
- **Grad-CAM**: Gradient-weighted Class Activation Mapping
- **Heatmap Overlay**: Visual explanation of model decisions
- **Focus Areas**: Highlights optic nerve and relevant regions

## Project Structure

```
AI-Glaucoma-Detection/
├── main.py                     # Main web interface (run this!)
├── requirements.txt            # Python dependencies
├── README.md                   
├── EfficientNetV2/             # Primary model (latest)
│   ├── best_glaucoma_model.pth
│   ├── model.ipynb
│   ├── efficientnetv2_training_results.png
│   ├── efficientnetv2_test_results.png
│   ├── efficientnetv2_validation_metrics.png
│   ├── test_results.json
│   └── train/val/test_split.csv
└── Backup Models/              # Previous model versions
    ├── EfficientNet/           # EfficientNet implementation
    │   ├── best_glaucoma_model.pth
    │   ├── model.ipynb
    │   ├── glaucoma_results.png
    │   ├── final_performance_comparison.png
    │   ├── Labels.csv
    │   └── Images/
    ├── ResNet50/               # ResNet50 implementation
    │   ├── best_glaucoma_model.pth
    │   ├── model.ipynb
    │   ├── glaucoma_training_results.png
    │   └── performance analysis files
    └── MobileNetV3-Large/      # MobileNetV3-Large implementation
        ├── best_glaucoma_model.pth
        ├── model.ipynb
        ├── glaucoma_results.png
        └── performance analysis files
```

## Usage Examples

### Command Line
```bash
# Start the web interface
python gradio_fixed.py

# The interface will be available at:
# - Local: http://localhost:7865
# - Public: https://[random-id].gradio.live
```

### Programmatic Usage
```python
from PIL import Image
import torch

# Load your trained model
model = load_model("EfficientNetV2/best_glaucoma_model.pth")

# Analyze an image
image = Image.open("retinal_image.jpg")
prediction, confidence, heatmap = predict_glaucoma(image)

print(f"Prediction: {prediction}")
print(f"Confidence: {confidence:.2%}")
```

## AI Methodology

1. **Image Preprocessing**: Resize, normalize, and prepare retinal images
2. **Feature Extraction**: EfficientNetV2 extracts hierarchical features
3. **Classification**: Binary classification (Normal vs Glaucoma)
4. **Explanation Generation**: Grad-CAM highlights decision regions
5. **Confidence Assessment**: Softmax probabilities for reliability

## Troubleshooting

### Common Issues

**Port Already in Use:**
```bash
# Kill process using port 7865
netstat -ano | findstr :7865
taskkill /PID [process_id] /F
```

**Dependency Conflicts:**
```bash
# Reinstall with correct versions
pip install -r requirements.txt --force-reinstall
```

**Model Loading Errors:**
- Ensure model files are in correct directories
- Check file permissions
- Verify PyTorch installation

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **PyTorch Team** for the deep learning framework
- **Gradio Team** for the web interface framework
- **Medical Community** for glaucoma research and datasets
- **Open Source Contributors** for various tools and libraries

## Contact

**Bharath Reddy Yedavalli**
- GitHub: [@BharathReddyYedavalli](https://github.com/BharathReddyYedavalli)
- Project: [AI-Glaucoma-Detection](https://github.com/BharathReddyYedavalli/AI-Glaucoma-Detection)

---

**Medical Disclaimer**: This tool is for research and educational purposes only. Always consult qualified medical professionals for clinical diagnosis and treatment decisions.
