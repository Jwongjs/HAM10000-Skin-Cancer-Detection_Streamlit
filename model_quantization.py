import torch
import timm
import streamlit as st

def load_model(model_path):
    """
    Load the pre-trained model from the specified path.
    Handles both regular checkpoints and quantized models.
    """
    try:
        # Initialize the model architecture (using timm)
        model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=7)

        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Handle checkpoint format load_state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Standard checkpoint format
            model.load_state_dict(checkpoint['model_state_dict'])

        # Set the model to evaluation mode
        model.eval()
        
        return model
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def load_quantized_model(model_path):
    """
    Load a quantized model from the specified path.
    """
    try:
        # Initialize the model architecture
        model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=7)
        
        # Quantize model
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        
        # Load the quantized state dict
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        quantized_model.load_state_dict(state_dict)
        
        # Set to evaluation mode
        quantized_model.eval()
        
        st.info("Quantized model loaded successfully!")
        return quantized_model
        
    except Exception as e:
        st.error(f"Error loading quantized model: {str(e)}")
        return None

# Load original model and quantize it
original_model = load_model("model/phase2_best.pth")

if original_model is not None:
    # Quantizes the model (reduces size significantly to accomodate streamlit deployment resource limit)
    quantized_model = torch.quantization.quantize_dynamic(
        original_model, {torch.nn.Linear}, dtype=torch.qint8
    )

    #Save the quantized model (just the state dict)
    torch.save(quantized_model.state_dict(), "model/phase2_best_quantized.pth")
    print("Quantized model saved successfully!")

    # Test loading the quantized model
    loaded_quantized_model = load_quantized_model("model/phase2_best_quantized.pth")
    if loaded_quantized_model is not None:
        print("Quantized model loaded and tested successfully!")
else:
    print("Failed to load original model for quantization")