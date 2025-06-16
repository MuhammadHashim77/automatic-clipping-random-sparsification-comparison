"""
    The code is a simple script that loads a pre-trained model and uses it to predict segmentation masks for the Duke OCT dataset.
    I plan to use this script to generate predictions for the test set and evaluate the performance of the model. 
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from networks import get_model
from torchvision import transforms


def preprocess_image(image_path, image_size=224):
    img = np.load(image_path)

    # Transform the image
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    img = transform(img)
    return img.unsqueeze(0)


def load_model(model_path, model_name="unet", in_channels=1, num_classes=9):
    model = get_model(model_name, in_channels=in_channels, num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    if list(state_dict.keys())[0].startswith("_module"):
        # Create a new state dict in which the key names do not contain the `_module` prefix
        state_dict = {k.replace("_module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


def predict(model, img):
    with torch.no_grad():
        output = model(img)
    return output


def plot_prediction(image, mask, prediction, file_name):
    # 3-panel view: input, ground truth, model output
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Input Image', fontsize=12)

    axs[1].imshow(mask, cmap='plasma')
    axs[1].set_title('True Segmentation', fontsize=12)

    axs[2].imshow(prediction, cmap='viridis')
    axs[2].set_title('Model Output', fontsize=12)

    # clean up axes
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(pad=2.0)
    plt.savefig(file_name, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Predict segmentation masks for Duke OCT dataset."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="code/unet.pt",
        help="Path to the trained model.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/DukeData/test",
        help="Path to the test data.",
    )
    args = parser.parse_args()

    model = load_model(
        args.model_path, model_name="unet", in_channels=1, num_classes=9
    ).to("cpu")
    image_files = os.listdir(os.path.join(args.data_path, "images"))

    if not os.path.exists("predictions"):
        os.makedirs("predictions")

    for file in image_files:
        image_path = os.path.join(args.data_path, "images", file)
        img = preprocess_image(image_path, image_size=224).to("cpu")
        img_tensor = torch.tensor(img, dtype=torch.float)

        prediction = predict(model, img_tensor)
        pred_mask = torch.argmax(prediction, dim=1).squeeze().numpy()
        mask = np.load(os.path.join(args.data_path, "masks", file))

        # Plot the original image, ground truth mask and predicted mask
        plot_prediction(
            img.squeeze(), mask, pred_mask, f"predictions/{file}_prediction.png"
        )


if __name__ == "__main__":
    main()
