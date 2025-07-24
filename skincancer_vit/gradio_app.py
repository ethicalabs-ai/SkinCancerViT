import traceback
import gradio as gr
import numpy as np
from PIL import Image
from datasets import load_dataset
import random

from skincancer_vit.model import SkinCancerViTModel
from skincancer_vit.xai_utils import (
    get_attention_map_output_gradcam,
    generate_cam_animation,
)
from skincancer_vit.utils import get_torch_device
from pytorch_grad_cam import EigenCAM

CAM_METHOD = EigenCAM


HF_MODEL_REPO = "ethicalabs/SkinCancerViT"
DEVICE = get_torch_device()
IMG_SIZE = (224, 224)  # Standard input size for ViT-base-patch16-224
PATCH_SIZE = 16  # Patch size for ViT-base-patch16

print(f"Loading SkinCancerViT model from {HF_MODEL_REPO} to {DEVICE}...")
model = SkinCancerViTModel.from_pretrained(HF_MODEL_REPO)
model.to(DEVICE)
model.eval()  # Set to evaluation mode
print("Model loaded successfully.")

print("Loading 'marmal88/skin_cancer' dataset for random samples...")
dataset = load_dataset("marmal88/skin_cancer", split="test")
print("Dataset loaded successfully.")


# --- Prediction Function for Uploaded Image ---
def predict_uploaded_image(
    image: Image.Image, age: int, localization: str
) -> tuple[str, np.ndarray]:
    """
    Handles prediction for an uploaded image with user-provided tabular data
    and generates an saliency map.
    """
    if model is None:
        return "Error: Model not loaded. Please check the console for details.", None
    if image is None:
        return "Please upload an image.", None
    if age is None:
        return "Please enter an age.", None
    if not localization:
        return "Please select a localization.", None

    try:
        # Call the model's full_predict method to get prediction and target class index
        predicted_dx, confidence = model.full_predict(
            raw_image=image,
            raw_age=age,
            raw_localization=localization,
            device=DEVICE,
        )
        predicted_class_idx = model.config.label2id[predicted_dx]

        # Generate saliency map using the vision_model part of your SkinCancerViTModel
        attention_map_image = get_attention_map_output_gradcam(
            full_multimodal_model=model,
            image_input=image,
            target_class_idx=predicted_class_idx,
            img_size=IMG_SIZE,
            cam_method=CAM_METHOD,
            patch_size=PATCH_SIZE,
            raw_age=age,
            raw_localization=localization,
            device=DEVICE,  # Pass device to xai_utils function
        )

        result_text = (
            f"Predicted Diagnosis: **{predicted_dx}** (Confidence: {confidence:.4f})"
        )
        return result_text, attention_map_image
    except Exception as e:
        traceback.print_exc()
        return f"Prediction Error: {e}", None


# --- Prediction Function for Random Sample ---
def predict_random_sample() -> tuple[Image.Image, str, np.ndarray]:
    """
    Fetches a random sample from the dataset, performs prediction,
    and generates an saliency map.
    """
    if model is None:
        return (
            None,
            "Error: Model not loaded. Please check the console for details.",
            None,
        )
    if dataset is None:
        return None, "Error: Dataset not loaded. Cannot select random sample.", None

    try:
        # Select a random sample from the dataset
        random_idx = random.randint(0, len(dataset) - 1)
        sample = dataset[random_idx]

        sample_image = sample["image"]
        sample_age = sample["age"]
        sample_localization = sample["localization"]
        sample_true_dx = sample["dx"]

        # Call the model's full_predict method
        predicted_dx, confidence = model.full_predict(
            raw_image=sample_image,
            raw_age=sample_age,
            raw_localization=sample_localization,
            device=DEVICE,
        )
        predicted_class_idx = model.config.label2id[predicted_dx]

        # Generate saliency map
        attention_map_image = get_attention_map_output_gradcam(
            full_multimodal_model=model,
            image_input=sample_image,
            target_class_idx=predicted_class_idx,
            img_size=IMG_SIZE,
            cam_method=CAM_METHOD,
            patch_size=PATCH_SIZE,
            raw_age=sample_age,
            raw_localization=sample_localization,
            device=DEVICE,
        )

        # Return a formatted string with all information
        result_str = (
            f"**Random Sample Details:**\n"
            f"- Age: {sample_age}\n"
            f"- Localization: {sample_localization}\n"
            f"- True Diagnosis: **{sample_true_dx}**\n\n"
            f"**Model Prediction:**\n"
            f"- Predicted Diagnosis: **{predicted_dx}**\n"
            f"- Confidence: {confidence:.4f}\n"
            f"- Correct Prediction: {'✅ Yes' if predicted_dx == sample_true_dx else '❌ No'}"
        )
        return sample_image, result_str, attention_map_image
    except Exception as e:
        traceback.print_exc()  # Print full traceback for debugging
        return None, f"Prediction Error on Random Sample: {e}", None


def predict_cam_animation(
    image: Image.Image, age: int, localization: str
) -> tuple[str, Image.Image]:
    try:
        predicted_dx, confidence, output_gif_path = generate_cam_animation(
            full_multimodal_model=model,
            image_input=image,
            age=age,
            localization=localization,
            img_size=IMG_SIZE,
            cam_method=CAM_METHOD,
            patch_size=PATCH_SIZE,
            output_gif_path="cam_animation_layer.gif",  # TODO: Unique name
        )
        result_str = (
            f"**Model Prediction:**\n"
            f"- Predicted Diagnosis: **{predicted_dx}**\n"
            f"- Confidence: {confidence:.4f}\n"
        )
        return result_str, output_gif_path
    except Exception as e:
        traceback.print_exc()  # Print full traceback for debugging
        return f"Prediction Error on Random Sample: {e}", None


# --- Gradio Interface ---
with gr.Blocks(title="Skin Cancer ViT Predictor") as demo:
    gr.Markdown(
        """
        # Skin Cancer ViT Predictor
        This application demonstrates the `SkinCancerViT` multimodal model for skin cancer diagnosis.
        It can take an uploaded image with patient metadata or predict on a random sample from the dataset.
        It also visualizes the saliency map generated by the Vision Transformer component.
        **Disclaimer:** This tool is for demonstration and research purposes only and should not be used for medical diagnosis.
        """
    )

    with gr.Tab("Predict on Random Sample"):
        gr.Markdown("## Get a Prediction from a Random Sample in the Test Set")
        random_sample_button = gr.Button("Get Random Sample Prediction")

        with gr.Row():
            output_random_image = gr.Image(
                type="pil", label="Random Sample Image", height=250, width=250
            )
            output_random_attention_map = gr.Image(
                type="numpy", label="Grad-CAM Saliency", height=250, width=250
            )
        output_random_details = gr.Markdown(
            "Random sample details and prediction will appear here."
        )

        random_sample_button.click(
            fn=predict_random_sample,
            inputs=[],
            outputs=[
                output_random_image,
                output_random_details,
                output_random_attention_map,
            ],
        )

    with gr.Tab("Upload Image & Predict"):
        gr.Markdown("## Upload Your Image and Provide Patient Data")
        with gr.Row():
            image_input = gr.Image(
                type="pil", label="Upload Skin Lesion Image (224x224 preferred)"
            )
            with gr.Column():
                age_input = gr.Number(
                    label="Patient Age", minimum=0, maximum=120, step=1
                )
                # Ensure these localizations match your training data categories
                localization_input = gr.Dropdown(
                    list(
                        model.config.localization_to_id.keys()
                    ),  # Convert dict_keys to list
                    label="Lesion Localization",
                    value="unknown",  # Default value
                )
                predict_button = gr.Button("Get Prediction")

        with gr.Row():
            output_upload = gr.Markdown("Prediction will appear here.")
            output_uploaded_attention_map = gr.Image(
                type="numpy", label="Grad-CAM Saliency", height=250, width=250
            )

        predict_button.click(
            fn=predict_uploaded_image,
            inputs=[image_input, age_input, localization_input],
            outputs=[output_upload, output_uploaded_attention_map],
        )

    with gr.Tab("Generate CAM Animation"):
        gr.Markdown("## Generate an Animation of Saliency Maps Across Layers")
        with gr.Row():
            anim_image_input = gr.Image(
                type="pil", label="Upload Skin Lesion Image for Animation"
            )
            with gr.Column():
                anim_age_input = gr.Number(
                    label="Patient Age", minimum=0, maximum=120, step=1
                )
                anim_localization_input = gr.Dropdown(
                    list(model.config.localization_to_id.keys()),
                    label="Lesion Localization",
                    value="unknown",
                )
                generate_anim_button = gr.Button("Generate Animation")

        with gr.Row():
            anim_output_gif = gr.Image(
                label="CAM Animation", type="filepath", format="gif", interactive=False
            )
            with gr.Column():
                anim_output_message = gr.Markdown("Prediction will appear here.")

        generate_anim_button.click(
            fn=lambda img, age, loc, sub_mod: predict_cam_animation(
                image=img,
                age=age,
                localization=loc,
            ),
            inputs=[
                anim_image_input,
                anim_age_input,
                anim_localization_input,
            ],
            outputs=[
                anim_output_message,
                anim_output_gif,
            ],  # Output the message and the downloadable file
        )

if __name__ == "__main__":
    demo.launch(share=False)
