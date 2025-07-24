import torch
from datasets import load_dataset
import os
import click
import pandas as pd
from datetime import datetime

from skincancer_vit.utils import get_torch_device
from skincancer_vit.model import SkinCancerViTModel


def load_model(model_path: str, device: torch.device) -> SkinCancerViTModel:
    """
    Loads the SkinCancerViTModel from the specified path.
    """
    # Check if the final model directory exists and contains necessary files
    if not os.path.exists(model_path) or not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Error: Final model directory '{model_path}' not found. "
            "Please ensure you have run main.py with the updated code to save the final model."
        )
    if not os.path.exists(os.path.join(model_path, "config.json")):
        raise FileNotFoundError(
            f"Error: 'config.json' not found in '{model_path}'. Please run train.py to save the model correctly."
        )
    if not os.path.exists(os.path.join(model_path, "model.safetensors")):
        raise FileNotFoundError(
            f"Error: 'model.safetensors' not found in '{model_path}'. Please run train.py to save the model correctly."
        )

    print(f"Loading model from: {model_path}")
    try:
        model = SkinCancerViTModel.from_pretrained(model_path)
        model.to(device)
        print(
            f"Trained model loaded successfully from '{model_path}' using from_pretrained."
        )
        return model
    except Exception as e:
        raise Exception(
            f"Error loading model from '{model_path}' using from_pretrained: {e}. "
            f"Ensure 'config.json' and 'model.safetensors' are present and valid."
        )


def run_inference(
    model: SkinCancerViTModel,
    device: torch.device,
    num_samples: int | None,  # num_samples can now be None
    output_csv_path: str,
):
    """
    Runs inference on a subset of the test dataset and saves results to a CSV file.

    Args:
        model (SkinCancerViTModel): The loaded model for inference.
        device (torch.device): The device (cuda or cpu) to perform inference on.
        num_samples (int | None): The number of test samples to run inference on.
                                  If None, all available samples will be used.
        output_csv_path (str): The path to save the inference results as a CSV file.
    """
    if num_samples is None:
        print("\nDemonstrating inference on ALL available test samples:")
    else:
        print(f"\nDemonstrating inference on {num_samples} test samples:")

    # Get examples from the test split
    try:
        dataset = load_dataset("marmal88/skin_cancer", split="test").shuffle(seed=42)
        if num_samples is not None:
            test_samples_raw = dataset.select(range(num_samples))
        else:
            test_samples_raw = dataset  # Select all samples if num_samples is None

        print(f"Successfully loaded {len(test_samples_raw)} test samples.")
    except Exception as e:
        print(
            f"Error loading dataset: {e}. Please check your internet connection or dataset name."
        )
        return

    prediction_results = []
    correct_predictions = 0
    total_samples_processed = 0

    for i, example in enumerate(test_samples_raw):
        pil_image = example["image"]
        age = example["age"]
        localization = example["localization"]
        true_dx = example["dx"]

        print(f"\n--- Sample {i + 1}/{len(test_samples_raw)} ---")
        print(
            f"Input: Age={age}, Localization='{localization}', True Diagnosis='{true_dx}'"
        )

        predicted_dx, predicted_confidence = model.full_predict(
            raw_image=pil_image,
            raw_age=age,
            raw_localization=localization,
            device=device,
        )
        predicted_dx_label = predicted_dx
        predicted_confidence_score = predicted_confidence

        is_correct = predicted_dx_label == true_dx
        if is_correct:
            correct_predictions += 1
            prediction_status = "CORRECT"
        else:
            prediction_status = "INCORRECT"

        print(
            f"Predicted Diagnosis: '{predicted_dx_label}' (Confidence: {predicted_confidence_score:.4f})"
        )
        print(f"Prediction: {prediction_status}")

        prediction_results.append(
            {
                "sample_id": i + 1,
                "true_diagnosis": true_dx,
                "predicted_diagnosis": predicted_dx_label,
                "confidence": predicted_confidence_score,
                "is_correct": is_correct,
                "age": age,
                "localization": localization,
            }
        )
        total_samples_processed += 1

    print("\nInference demonstration complete.")

    # Calculate and print accuracy
    if total_samples_processed > 0:
        accuracy_percentage = (correct_predictions / total_samples_processed) * 100
        print(
            f"Demonstration Accuracy: {accuracy_percentage:.2f}% ({correct_predictions}/{total_samples_processed} correct)"
        )
    else:
        print("No samples were processed for demonstration.")

    # Save results to CSV
    if prediction_results:
        df = pd.DataFrame(prediction_results)
        df.to_csv(output_csv_path, index=False)
        print(f"Inference results saved to '{output_csv_path}'")
    else:
        print("No prediction results to save.")


@click.command()
@click.option(
    "--model_handle_or_path",
    default="./results/final_model",
    help="Path to the directory containing the trained model (config.json, model.safetensors).",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.option(
    "--output_csv_path",
    default=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    help="Path to save the inference results CSV file.",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
)
@click.option(
    "--num_samples",
    default=None,
    type=int,
    help="Number of test samples to run inference on. If not specified, all records are selected.",
)
def main(model_handle_or_path: str, output_csv_path: str, num_samples: int | None):
    """
    CLI tool for performing skin cancer diagnosis inference.

    Loads a trained SkinCancerViTModel and runs predictions on a subset of the
    skin cancer test dataset, saving the results to a CSV file.
    """
    print("Starting skin cancer inference script...")

    device = get_torch_device()
    print(f"Using device: {device}")

    try:
        model = load_model(model_handle_or_path, device)
        run_inference(model, device, num_samples, output_csv_path)
    except FileNotFoundError as fnf_e:
        print(f"Error: {fnf_e}")
        print(
            "Please ensure the model path is correct and contains all necessary files."
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print("\nScript finished.")


if __name__ == "__main__":
    main()
