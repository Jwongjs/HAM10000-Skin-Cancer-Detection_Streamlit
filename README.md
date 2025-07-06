# Skin Cancer Image Classification Web App
**Deployed Streamlit Web App uses the dummy version of the image classification model due to resource limitations

A Streamlit-based web application that uses deep learning to classify skin lesions into seven different categories, helping with early detection of potential skin cancer conditions.

## 🌟 Features

- **AI-Powered Classification**: Uses EfficientNet-B0 deep learning model for accurate skin lesion classification
- **7 Lesion Categories**: Classifies images into melanoma, basal cell carcinoma, melanocytic nevi, and 4 other types
- **Risk Assessment**: Provides medical suggestions based on confidence thresholds
- **Interactive Interface**: User-friendly Streamlit interface with image upload and analysis
- **Real-time Processing**: Instant classification results with confidence scores

## 🏥 Supported Lesion Types

1. **nv (Melanocytic nevi)** - Common moles, typically benign
2. **mel (Melanoma)** - Serious skin cancer requiring immediate attention
3. **bkl (Benign keratosis-like lesions)** - Non-cancerous growths
4. **bcc (Basal cell carcinoma)** - Most common but least aggressive skin cancer
5. **akiec (Actinic keratoses)** - Precancerous patches from sun damage
6. **vasc (Vascular lesions)** - Abnormal blood vessel growths
7. **df (Dermatofibroma)** - Benign skin nodules

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Jwongjs/HAM10000-Skin-Cancer-Detection_Streamlit.git
cd HAM10000-Skin-Cancer-Detection_Streamlit
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the pre-trained model:
   - Place your trained model file (`phase2_best.pth`) in the `model/` directory

4. Run the application:
```bash
streamlit run mainpage.py
```

5. Open your browser and navigate to `http://localhost:8501`

## 📋 Requirements

```txt
streamlit
streamlit-lottie
torch
torchvision
timm
Pillow
numpy
```

## 🏗️ Project Structure

```
skin-cancer-classification/
├── mainpage.py                 # Main Streamlit application
├── lottie_animations.py        # Animation handler
├── requirements.txt            # Python dependencies
├── model/                      # Model files directory
│   ├── phase1_best.pth
│   ├── phase1_best(undersampled).pth
│   ├── phase2_best.pth
│   └── phase2_best(undersampled).pth
├── lottie_animations/          # Animation assets
│   └── mainpage.json
└── README.md
```

## 🔬 How It Works

1. **Image Upload**: Users upload skin lesion images (JPG, JPEG, PNG)
2. **Preprocessing**: Images are resized, normalized using training statistics
3. **Classification**: EfficientNet-B0 model processes the image
4. **Risk Assessment**: 
   - Melanoma threshold: 20% confidence
   - BCC/AKIEC threshold: 30% confidence
5. **Results Display**: Shows classification, confidence scores, and medical advice

## ⚕️ Medical Disclaimers

> **Important**: This application is for educational and research purposes only. It should NOT be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical advice and diagnosis.

- Results are not definitive medical diagnoses
- High-risk predictions require immediate dermatologist consultation
- Regular skin examinations by healthcare providers are recommended

## 🛠️ Model Information

- **Architecture**: EfficientNet-B0
- **Training**: Two-phase training approach
- **Classes**: 7 skin lesion categories
- **Input Size**: 224x224 pixels
- **Normalization**: Custom statistics from training dataset

## 🎯 Usage Tips

1. **Image Quality**: Use clear, well-lit images for best results
2. **File Formats**: Supports JPG, JPEG, and PNG formats
3. **Image Size**: Any size (automatically resized to 224x224)
4. **Medical Advice**: Pay attention to threshold warnings for high-risk conditions

## 🙏 Acknowledgments

- HAM10000 dataset for training data
- EfficientNet architecture by Google Research
- Streamlit for the web framework
- Medical professionals for guidance on classification categories

## 📊 Performance Metrics

- Model validation accuracy: Displayed when model loads
- Real-time confidence scores for all classifications
- Threshold-based risk assessment system

---

**⚠️ Medical Disclaimer**: This tool is for educational purposes only. Consult healthcare professionals for medical advice.
