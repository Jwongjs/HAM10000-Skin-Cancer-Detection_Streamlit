##Official Model not integrated, dummy model is used due to streamlit resource limitation

import streamlit as st 
from streamlit_lottie import st_lottie
import lottie_animations as l
from PIL import Image
from torchvision import transforms
import timm
import torch
from pathlib import Path
import requests

st.set_page_config(page_title="Skin Cancer Image Classification", page_icon="🧑‍⚕️", layout="wide")

def page_intro_section():
    ##Page introduction section            
    st.title("Skin Cancer Image Classification")

    col1, col2 = st.columns([1,1])
    with col1:
        st.subheader("🧠 Detect the Signs Early with AI-Powered Precision")
        st.write("Upload skin images and let advanced deep learning models assist in identifying potential cancerous conditions with confidence and speed.")
        with st.expander("🩺 Explore the Skin Lesion Categories"):
            st.write("- **nv (Melanocytic nevi)**: Commonly known as moles, these are typically benign growths caused by clusters of pigmented cells.")
            st.write("- **mel (Melanoma)**: A serious form of skin cancer that originates in melanocytes; early detection is critical for treatment.")
            st.write("- **bkl (Benign keratosis-like lesions)**: Non-cancerous growths that may resemble warts or seborrheic keratoses.")
            st.write("- **bcc (Basal cell carcinoma)**: A type of skin cancer that begins in the basal cells; it's the most common but least aggressive form.")
            st.write("- **akiec (Actinic keratoses)**: Precancerous patches caused by sun damage that can potentially develop into squamous cell carcinoma.")
            st.write("- **vasc (Vascular lesions)**: Abnormal growths of blood vessels, usually benign, but sometimes mistaken for malignancies.")
            st.write("- **df (Dermatofibroma)**: Benign skin nodules often caused by minor injuries like insect bites or ingrown hairs.")
    with col2:
        st_lottie(l.mainpage_lottie, loop = True, width = 260, height = 250, key = None)
        
def load_model(model_path):
    """
    Load the pre-trained model from the specified path.
    """
    try:
        # Initialize the model architecture (using timm)
        model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=7)  # Assume 7 classes

        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        # Load just the model state dict from the checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])

        # Set the model to evaluation mode
        model.eval()
        
        # Optional: Print validation accuracy from checkpoint
        if 'val_acc' in checkpoint:
            st.info(f"Model's validation accuracy: {checkpoint['val_acc']:.4f}")
        
        return model
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None
        
def get_standard_transform():
    #Same image transformation stats from training notebook for efficientNet_b0 model
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.7630331, 0.5456457, 0.5700467],  
            std=[0.1409281, 0.15261227, 0.16997086]     
        )
    ])

def validate_image_file(image_file):
    """
    Validates the uploaded image file and preprocesses it for model input
    Returns: (is_valid, message, processed_image)
    """
    if image_file is None:
        return False, "Please upload an image file", None
    
    try:
        #Read and validate image
        image = Image.open(image_file)
        
        #Convert to RGB if needed
        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
            image = image.convert('RGB')
        
        #Apply preprocessing transforms
        transform = get_standard_transform()
        processed_image = transform(image)
        
        # Add batch dimension
        #To prepare the image tensor for input into deep learning models.
        processed_image = processed_image.unsqueeze(0)
        
        return True, "Image successfully processed", processed_image
        
    except Exception as e:
        return False, f"Error processing image: {str(e)}", None

def confidence_score_evaluation(prediction):
    # Display confidence scores and advice for each class
    confidence_scores = torch.nn.functional.softmax(prediction, dim=1)[0] * 100
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("💡 Confidence Scores")
        for i, score in enumerate(confidence_scores):
            label = class_labels.get(i, "Unknown")
            st.write(f"{label}: {score.item():.2f}%")
    with col2:
        st.subheader("🩺 Medical Suggestions")                    

        MELANOMA_THRESHOLD = 20.0  #20% confidence for melanoma
        BCC_AKIEC_THRESHOLD = 30.0  #30% confidence for BCC/AKIEC
        
        #Threshold pass checker
        warnings_shown = False
        
        #Check for melanoma (index 1)
        melanoma_score = confidence_scores[1].item()
        if melanoma_score >= MELANOMA_THRESHOLD:
            st.error(f"⚠️ **ATTENTION:** Melanoma confidence ({melanoma_score:.1f}%) exceeds the safety threshold")
            st.write("##### **Immediate dermatologist consultation is strongly recommended due to high risk.**")
            warnings_shown = True

        #Check for both BCC (index 3) and AKIEC (index 4)
        bcc_score = confidence_scores[3].item()
        akiec_score = confidence_scores[4].item()
        if bcc_score >= BCC_AKIEC_THRESHOLD or akiec_score >= BCC_AKIEC_THRESHOLD:
            st.warning(f"⚠️ High confidence in potentially serious conditions:")
            if bcc_score >= BCC_AKIEC_THRESHOLD:
                st.write(f"- Basal Cell Carcinoma: {bcc_score:.1f}%")
            if akiec_score >= BCC_AKIEC_THRESHOLD:
                st.write(f"- Actinic Keratoses: {akiec_score:.1f}%")
            st.write("**Regular monitoring and consultation with a dermatologist is recommended.**")
            warnings_shown = True
            
        #If there are no thresholds exceeded, then general advice will be shown
        if not warnings_shown:
            st.info("✅ No high-risk indicators detected")
            st.write("**General Recommendations:**")
            st.write("- Continue regular skin self-examinations")
            st.write("- Practice sun safety:")
            st.write("  - Use broad-spectrum sunscreen")
            st.write("  - Wear protective clothing")
            st.write("  - Avoid peak UV hours")
            st.write("- Document any changes in existing skin lesions")
            st.write("- Consider annual skin checks with a healthcare provider")

page_intro_section()
st.divider()
image_file = st.file_uploader("#### Classify the type of your skin lesion (with EfficientNet-b0) ⬇️", type=["jpg", "jpeg", "png"])

#Load the best fine-tuned model after running main2(official_with_weightedSampling).ipynb
model_path = "model/phase2_best.pth"
model = load_model(model_path)

button_clicked = st.button("🔍 Analyze Image")
if button_clicked:
    is_valid, message, processed_image = validate_image_file(image_file)
    if not is_valid:
        st.error(message)
    else:
        with st.spinner('Analyzing image...'):
            try:
                #Display the original image
                image = Image.open(image_file)
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:  
                    st.image(image, caption='Uploaded Image', width=400)
                
                #Skin Cancer prediction
                prediction = model(processed_image)
                
                ##Process prediction and results display
                # Get the index of the highest predicted class
                _, predicted_class = torch.max(prediction, 1)
                predicted_index = predicted_class.item()

                # Define label names
                class_labels = {
                    0: 'Melanocytic nevi (nv)',
                    1: 'Melanoma (mel)',
                    2: 'Benign keratosis-like lesions (bkl)',
                    3: 'Basal cell carcinoma (bcc)',
                    4: 'Actinic keratoses (akiec)',
                    5: 'Vascular lesions (vasc)',
                    6: 'Dermatofibroma (df)'
                }

                # Display prediction
                predicted_label = class_labels.get(predicted_index, "Unknown")
                st.success(f"**Prediction:** {predicted_label}")
                
                # Display confidence scores and medical suggestions
                confidence_score_evaluation(prediction)

            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
        
