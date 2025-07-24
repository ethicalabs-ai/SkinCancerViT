import torch
import torch.nn as nn
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
import cv2
import torchvision.transforms as transforms

from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from skincancer_vit.utils import get_torch_device


def preprocess_image_for_vit(
    image: Image.Image, img_size: tuple, device: torch.device
) -> torch.Tensor:
    """
    Preprocesses a PIL Image for the Vision Transformer model's input.
    Applies resizing, conversion to tensor, and standard ImageNet normalization.

    Args:
        image (PIL.Image.Image): The input image.
        img_size (tuple): Target image size (width, height) for the model's input.
        device (torch.device): The device to move the tensor to (e.g., 'cuda' or 'cpu').
    Returns:
        torch.Tensor: The preprocessed image tensor, ready for model input (with batch dimension).
                      `requires_grad` is set to True, which is necessary for gradient-based
                      attribution methods.
    """
    transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Standard ImageNet normalization
        ]
    )
    # Add batch dimension (B, C, H, W) and ensure gradients can be computed.
    input_tensor = transform(image).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)  # Essential for gradient-based methods
    return input_tensor


def overlay_heatmap_on_image(
    original_image_np: np.ndarray, heatmap_np: np.ndarray
) -> np.ndarray:
    """
    Overlays a grayscale heatmap (typically 0-1 range) onto an original RGB image.

    Args:
        original_image_np (np.ndarray): The original image as a NumPy array (H_orig, W_orig, 3) in [0, 255] range.
                                        Assumed to be in RGB format.
        heatmap_np (np.ndarray): The heatmap as a NumPy array (H_cam, W_cam) in [0, 1] range.
    Returns:
        np.ndarray: The overlaid image as a NumPy array (H_orig, W_orig, 3) in [0, 255] range,
                    suitable for display.
    """
    # Resize heatmap to match original image dimensions.
    # cv2.resize expects (width, height) for dsize.
    # original_image_np.shape is (H, W, C), so width is shape[1] and height is shape[0].
    heatmap_resized = cv2.resize(
        heatmap_np, (original_image_np.shape[1], original_image_np.shape[0])
    )

    # Convert original image to float [0,1] for blending
    original_image_float = original_image_np.astype(np.float32) / 255

    # show_cam_on_image expects the image to be float [0,1] and the CAM to be float [0,1].
    # It will handle colormapping the CAM and overlaying it.
    # use_rgb=True is crucial here because original_image_float is an RGB image.
    visualization = show_cam_on_image(
        original_image_float, heatmap_resized, use_rgb=True
    )

    return (visualization * 255).astype(np.uint8)


def vit_reshape_transform(
    tensor: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """
    A reshape transform function specifically for Vision Transformers.
    It removes the class token ([:, 0, :]) and reshapes the remaining
    patch tokens into a 2D spatial grid, then transposes for correct
    channel ordering for CAM computation.

    Args:
        tensor (torch.Tensor): The output tensor from a ViT layer, typically
                               of shape (batch_size, num_patches + 1, channels).
        height (int): The height of the patch grid (e.g., 224 / 16 = 14).
        width (int): The width of the patch grid (e.g., 224 / 16 = 14).
    Returns:
        torch.Tensor: The reshaped tensor ready for CAM computation,
                      of shape (batch_size, channels, height, width).
    """
    # Exclude the class token (the first token)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    # Permute to (batch_size, channels, height, width)
    result = result.transpose(2, 3).transpose(1, 2)
    return result


class SkinCancerViTWrapperForCAM(nn.Module):
    """
    A wrapper around SkinCancerViTModel to make it compatible with pytorch-grad-cam.
    It takes only pixel_values as input and uses a fixed tabular_features tensor
    to produce the final classification logits. This allows Grad-CAM to focus
    on the image contribution while still using the full model's decision path.
    """

    def __init__(self, original_model: nn.Module, fixed_tabular_features: torch.Tensor):
        super().__init__()
        self.original_model = original_model
        self.fixed_tabular_features = fixed_tabular_features

    def forward(self, pixel_values: torch.Tensor):
        # Get the batch size from the image tensor. This is crucial because methods
        # like ScoreCAM pass a batch of perturbed images, not a single image.
        batch_size = pixel_values.size(0)

        # Ensure the fixed_tabular_features are on the same device as pixel_values
        fixed_tabular_features_on_device = self.fixed_tabular_features.to(
            pixel_values.device
        )

        # Repeat the tabular features to match the batch size of the image tensor.
        # The original tabular features have a batch size of 1.
        # .expand() is efficient as it doesn't copy data, it just creates a view.
        expanded_tabular_features = fixed_tabular_features_on_device.expand(
            batch_size, -1
        )
        # The -1 tells expand to not change the size of that dimension.

        # Call the original multimodal model's forward pass
        # with tensors that now have matching batch sizes.
        outputs = self.original_model(
            pixel_values=pixel_values, tabular_features=expanded_tabular_features
        )
        return outputs["logits"]


def get_attention_map_output_gradcam(
    full_multimodal_model: torch.nn.Module,  # Full SkinCancerViTModel
    raw_age: int,  # Raw age for tabular features
    raw_localization: str,  # Raw localization for tabular features
    image_input: Image.Image,
    target_class_idx: int,
    img_size: tuple,
    cam_method: type,
    patch_size: int,
    device: torch.device,
    target_layer_index: int = 5,
) -> np.ndarray:
    """
    Generates an attention map using pytorch-grad-cam for a specific class prediction
    and overlays it on the original image.

    Args:
        full_multimodal_model (torch.nn.Module): The full SkinCancerViTModel.
        raw_age (int): The raw age for tabular feature generation.
        raw_localization (str): The raw localization for tabular feature generation.
        image_input (PIL.Image.Image): The input image.
        target_class_idx (int): The index of the class for which to generate the explanation.
        img_size (tuple): Target image size (width, height) for the model's input.
        cam_method (type): The CAM method class to use (e.g., GradCAM, GradCAMPlusPlus).
        patch_size (int): The patch size used by the ViT model.
        device (torch.device): The device to run the CAM computation on.

    Returns:
        np.ndarray: The original image with the attention map overlaid, as a NumPy array.
    """
    full_multimodal_model.eval()  # Ensure the full model is in evaluation mode

    # Determine the actual device to use
    device = get_torch_device()

    # Move the full model to the target device
    full_multimodal_model.to(device)

    # Preprocess the image for the model's input (only pixel_values are needed for the wrapper's forward)
    input_tensor = preprocess_image_for_vit(image_input, img_size, device)

    # Generate fixed tabular features for the CAM wrapper
    # This part replicates the tabular preprocessing from SkinCancerViTModel's full_predict
    localization_one_hot = torch.zeros(
        full_multimodal_model.config.num_localization_features, device=device
    )
    if raw_localization in full_multimodal_model.config.localization_to_id:
        localization_one_hot[
            full_multimodal_model.config.localization_to_id[raw_localization]
        ] = 1.0

    def normalize_age_func_reconstructed_for_cam(age_value, config):
        # Replicates the age normalization logic from the model's full_predict
        if age_value is None:
            return (config.age_mean - config.age_min) / (
                config.age_max - config.age_min
            )
        if (config.age_max - config.age_min) == 0:
            return 0.0  # Handle division by zero if min and max are the same
        return (age_value - config.age_min) / (config.age_max - config.age_min)

    age_normalized_value = normalize_age_func_reconstructed_for_cam(
        raw_age, full_multimodal_model.config
    )
    age_normalized = torch.tensor(
        [age_normalized_value], dtype=torch.float, device=device
    )

    fixed_tabular_features = torch.cat(
        [localization_one_hot, age_normalized]
    ).unsqueeze(0)  # Add batch dimension

    # Create the wrapper model for CAM
    cam_wrapper_model = SkinCancerViTWrapperForCAM(
        full_multimodal_model, fixed_tabular_features
    ).to(device)

    # Get the original image as a NumPy array (0-255 RGB) for overlaying
    original_image_np = np.array(image_input.convert("RGB"))

    # Determine target layer for the ViT model (within the original_model's vision_model)
    # For ViT, a common target layer is the last LayerNorm after the attention block
    # or the MLP block in the encoder. 'layernorm_after' is a good choice as it's
    # the output of the entire encoder layer.
    target_layers = [
        cam_wrapper_model.original_model.vision_model.encoder.layer[
            target_layer_index
        ].layernorm_after,
    ]
    print(f"Using target_layers for CAM: {target_layers}")

    # Calculate spatial dimensions for reshape_transform
    grid_height = img_size[0] // patch_size
    grid_width = img_size[1] // patch_size
    if (
        grid_height * patch_size != img_size[0]
        or grid_width * patch_size != img_size[1]
    ):
        print(
            f"Warning: Image size {img_size} is not perfectly divisible by patch size {patch_size}."
        )
        print(
            f"Calculated grid: {grid_height}x{grid_width}. Actual image size: {img_size[0]}x{img_size[1]}"
        )

    # Prepare targets for CAM (explain a specific class prediction)
    targets = [ClassifierOutputTarget(target_class_idx)]

    # Initialize the CAM object with the wrapper model
    with cam_method(
        model=cam_wrapper_model,  # Pass the wrapper model
        target_layers=target_layers,  # target_layers expects a list
        reshape_transform=lambda x: vit_reshape_transform(
            x, height=grid_height, width=grid_width
        ),
    ) as cam:
        # Generate the grayscale CAM
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets,
            aug_smooth=True,
            eigen_smooth=True,
        )
        grayscale_cam = grayscale_cam[
            0, :
        ]  # Remove batch dimension (assuming batch_size=1)
        print("Grayscale CAM stats:")
        print(f"  - Min: {grayscale_cam.min():.6f}")
        print(f"  - Max: {grayscale_cam.max():.6f}")
        print(f"  - Mean: {grayscale_cam.mean():.6f}")
        print(f"  - Std Dev: {grayscale_cam.std():.6f}")
        # Overlay the generated CAM on the original image using the dedicated function
        enriched_image = overlay_heatmap_on_image(original_image_np, grayscale_cam)

        return enriched_image


def generate_cam_animation(
    full_multimodal_model: torch.nn.Module,  # Full SkinCancerViTModel
    image_input: Image.Image,
    age: int,
    localization: str,
    img_size: tuple,
    cam_method: type,
    patch_size: int,
    output_gif_path: str = "cam_animation.gif",
    duration_per_frame: float = 0.5,  # seconds
):
    """
    Generates an animation of Grad-CAM saliency maps across ViT encoder layers.

    Args:
        image_input (PIL.Image.Image): The input image.
        age (int): Patient age.
        localization (str): Lesion localization.
        output_gif_path (str): Path to save the output GIF.
        duration_per_frame (float): Duration for each frame in the GIF.
        target_sub_module_name (str): The specific sub-module within each layer to target.
    """
    print(f"Generating CAM animation for {output_gif_path}...")

    # Predict the target class once for consistency
    predicted_dx, confidence = full_multimodal_model.full_predict(
        raw_image=image_input,
        raw_age=age,
        raw_localization=localization,
        device=get_torch_device(),
    )
    predicted_class_idx = full_multimodal_model.config.label2id[predicted_dx]
    print(f"Predicted class for animation: {predicted_dx} (ID: {predicted_class_idx})")

    frames = []
    num_encoder_layers = len(full_multimodal_model.vision_model.encoder.layer)

    # Load a font for text overlay (adjust path/font as needed)
    try:
        font = ImageFont.truetype("arial.ttf", 20)  # Common font on Windows/Linux
    except IOError:
        print("Could not load arial.ttf, using default font.")
        font = ImageFont.load_default()

    for i in range(int(num_encoder_layers)):
        print(f"Processing layer {i + 1}/{num_encoder_layers}...")
        cam_image_np = get_attention_map_output_gradcam(
            full_multimodal_model=full_multimodal_model,
            raw_age=age,
            raw_localization=localization,
            image_input=image_input,
            target_class_idx=predicted_class_idx,
            img_size=img_size,
            cam_method=cam_method,
            patch_size=patch_size,
            device=get_torch_device(),
            target_layer_index=i,  # Pass the current layer index
        )

        # Convert numpy array back to PIL Image to add text
        cam_pil_image = Image.fromarray(cam_image_np)
        draw = ImageDraw.Draw(cam_pil_image)

        # Add text overlay: Layer number and predicted class
        text = f"Layer {i + 1}/{int(num_encoder_layers)} - Pred: {predicted_dx}"
        text_color = (255, 255, 255)  # White color
        # Add a black outline for better readability
        draw.text(
            (10, 10),
            text,
            font=font,
            fill=(0, 0, 0),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
        draw.text((10, 10), text, font=font, fill=text_color)

        frames.append(cam_pil_image)

    duration = int(1000 * duration_per_frame)
    # Save the frames as a GIF
    # https://github.com/imageio/imageio/issues/973
    imageio.mimsave(
        output_gif_path, frames, duration=duration, loop=0
    )  # loop=0 means infinite loop
    print(f"Animation saved to {output_gif_path}")
    return predicted_dx, confidence, output_gif_path
