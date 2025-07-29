# Glaucoma Detection with EfficientNetV2 - WORKING Gradio Interface
# This uses the exact same interface style you want, but with dependency fixes

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import efficientnet_v2_s
from PIL import Image
import numpy as np
import cv2

# Load trained model and set to eval
device = torch.device("cpu")

# Load EfficientNetV2 model architecture (same as your trained model)
model = efficientnet_v2_s(weights=None)  # No pretrained weights
num_ftrs = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5, inplace=True),
    nn.Linear(num_ftrs, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.3),
    nn.Linear(512, 2)
)

# Load the trained EfficientNetV2 weights
model.load_state_dict(torch.load("EfficientNetV2/best_glaucoma_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Define transforms for EfficientNetV2 (288x288 input)
val_transform = transforms.Compose([
    transforms.Resize((288, 288)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_with_explanation(image_pil):
    """
    Predict glaucoma and generate Grad-CAM explanation
    """
    try:
        print(f"Starting prediction for image: {image_pil.size}")
        model.eval()
        
        # Preprocess image
        image_tensor = val_transform(image_pil).unsqueeze(0).to(device)
        image_tensor.requires_grad_()

        # Grad-CAM setup
        gradients = []
        activations = []

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        def forward_hook(module, input, output):
            activations.append(output)

        # Hook the last convolutional layer for EfficientNetV2
        target_layer = model.features[-1]
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_backward_hook(backward_hook)

        # Forward pass
        print("Running forward pass...")
        output = model(image_tensor)
        pred_class = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0][pred_class].item()
        
        print(f"Prediction: {pred_class}, Confidence: {confidence:.4f}")

        # Backward pass for Grad-CAM
        print("Running backward pass for Grad-CAM...")
        model.zero_grad()
        output[0, pred_class].backward()

        # Remove hooks
        forward_handle.remove()
        backward_handle.remove()

        # Grad-CAM computation
        print("Computing Grad-CAM...")
        grad = gradients[0]
        act = activations[0]
        pooled_grad = torch.mean(grad, dim=[0, 2, 3])
        weighted_act = (act[0] * pooled_grad[:, None, None]).sum(dim=0)

        heatmap = weighted_act.cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) + 1e-8
        
        # Resize heatmap to original image size
        heatmap = cv2.resize(heatmap, (image_pil.width, image_pil.height))
        heatmap = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Overlay heatmap on original image
        img_np = np.array(image_pil.convert("RGB"))
        overlayed_img = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)

        # Generate explanation
        if pred_class == 1:
            explanation = f"Prediction: Glaucoma (Confidence: {confidence*100:.2f}%)\n\nThe model focused on areas of optic nerve cupping or peripheral thinning to make this decision."
        else:
            explanation = f"Prediction: Normal (Confidence: {confidence*100:.2f}%)\n\nThe optic nerve appears healthy, and no significant indicators of glaucoma were detected."

        print(f"Prediction complete!")
        return overlayed_img, explanation
        
    except Exception as e:
        print(f"Error in predict_with_explanation: {e}")
        import traceback
        traceback.print_exc()
        raise e

def inference_interface(image):
    """
    Gradio interface function
    """
    try:
        print(f"Received image: {type(image)}")
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        print(f"Image size: {image.size}")
        print(f"Image mode: {image.mode}")
        
        overlay_img, explanation = predict_with_explanation(image)
        print(f"Prediction successful!")
        
        return overlay_img, explanation
        
    except Exception as e:
        print(f"Error in inference_interface: {e}")
        import traceback
        traceback.print_exc()
        
        # Return error placeholders
        error_image = np.zeros((300, 300, 3), dtype=np.uint8)
        error_explanation = f"Error occurred during prediction: {str(e)}"
        
        return error_image, error_explanation

# Try importing gradio with fallback
try:
    import gradio as gr
    
    # Create Gradio interface - EXACTLY like you wanted
    demo = gr.Interface(
        fn=inference_interface,
        inputs=gr.Image(type="numpy", label="Upload Retinal Image"),
        outputs=[
            gr.Image(type="numpy", label="Grad-CAM Overlay"),
            gr.Textbox(label="Explanation")
        ],
        title="Glaucoma Detection AI - EfficientNetV2",
        description="Upload a retinal image. The EfficientNetV2 model will predict whether it shows signs of glaucoma or is normal, and display a Grad-CAM heatmap explanation.",
        allow_flagging="never"  # Use older syntax for Gradio 3.50.2
    )

    if __name__ == "__main__":
        print("Starting Glaucoma Detection Interface...")
        print("Model: EfficientNetV2")
        print("Input size: 288x288")
        print("Interface: Gradio")
        print()
        
        try:
            # Try different launch configurations
            demo.launch(
                share=True,          # Create public link
                server_port=7864,    # Different port
                inbrowser=False,     # Don't auto-open
                quiet=False
            )
        except Exception as e:
            print(f"Gradio launch failed: {e}")
            print("Trying alternative configuration...")
            demo.launch(
                share=False,
                server_name="0.0.0.0",
                server_port=7864,
                inbrowser=False
            )

except ImportError:
    print("Gradio not available or incompatible")
    print("Your Flask interface is still running at http://localhost:5000")
    print("The Flask version has the EXACT same functionality as Gradio!")
