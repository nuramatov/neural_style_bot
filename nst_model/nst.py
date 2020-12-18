from PIL import Image
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms, models

torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cnn = models.vgg19(pretrained=True).features.to(device).eval().requires_grad_(False)

# replace maxpools with avgpool

for (i, layer) in enumerate(cnn):
    if isinstance(layer, torch.nn.MaxPool2d):
        cnn[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)


def preprocess(image, max_size=None, shape=None):
    if max_size is None:
        max_size = 128
    image = image.convert("RGB")

    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    in_transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    # discard the alpha channel (that's the :3) and add the batch dimension

    image = in_transform(image)[:3, :, :].unsqueeze(0)

    return image


def im_convert(tensor):

    # convert tesnor to image, denormalize

    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image


def get_features(image, model, layers=None):

    if layers is None:
        layers = {
            "0": "conv1_1",
            "5": "conv2_1",
            "10": "conv3_1",
            "19": "conv4_1",
            "21": "conv4_2",  # content layer
            "28": "conv5_1",
        }

    features = {}
    x = image

    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features


def gram_matrix(tensor):

    batch_size, channels, height, width = tensor.size()

    tensor = tensor.view(batch_size * channels, height * width)

    gram = tensor @ tensor.t()

    return gram


def NST(
    content,
    style,
    style_2=None,
    num_epochs=None,
    style_weight=None,
    style_2_weight=None,
    max_size=None,
):
    content = preprocess(content, max_size=max_size).to(device)
    content_features = get_features(content, cnn)

    style = preprocess(style, shape=content.shape[-2:]).to(device)
    style_features = get_features(style, cnn)
    style_grams = {
        layer: gram_matrix(style_features[layer]) for layer in style_features
    }

    if style_2 is not None:
        style_2 = preprocess(style_2, shape=content.shape[-2:]).to(device)
        style_2_features = get_features(style_2, cnn)
        style_2_grams = {
            layer: gram_matrix(style_2_features[layer]) for layer in style_2_features
        }

    target = content.clone().requires_grad_(True).to(device)

    style_weights = {
        "conv1_1": 1.0,
        "conv2_1": 0.75,
        "conv3_1": 0.2,
        "conv4_1": 0.2,
        "conv5_1": 0.2,
    }

    style_2_weights = {
        "conv1_1": 1.0,
        "conv2_1": 0.75,
        "conv3_1": 0.2,
        "conv4_1": 0.2,
        "conv5_1": 0.2,
    }

    content_weight = 1
    if style_weight is None:
        style_weight = 1e9
    if style_2_weight is None:
        style_2_weight = 1e9

    optimizer = optim.Adam([target], lr=3e-3)

    if num_epochs is None:
        num_epochs = 500

    for epoch in range(1, num_epochs + 1):

        target_features = get_features(target, cnn)

        content_loss = torch.mean(
            (target_features["conv4_2"] - content_features["conv4_2"]) ** 2
        )

        style_loss = 0
        style_2_loss = 0

        for layer in style_weights:
            # get the "target" style representation for the layer
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            _, d, h, w = target_feature.shape
            # get the "style" style representation
            style_gram = style_grams[layer]
            # the style loss for one layer, weighted appropriately
            layer_style_loss = style_weights[layer] * torch.mean(
                (target_gram - style_gram) ** 2
            )
            # add to the style loss
            style_loss += layer_style_loss / (d * h * w)

            if style_2 is not None:
                style_2_gram = style_2_grams[layer]
                layer_style_2_loss = style_2_weights[layer] * torch.mean(
                    (target_gram - style_2_gram) ** 2
                )
                style_2_loss += layer_style_2_loss / (d * h * w)

        total_loss = (
            content_weight * content_loss
            + style_weight * style_loss
            + style_2_weight * style_2_loss
        )

        # update your target image
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    return Image.fromarray(np.uint8(im_convert(target) * 255))


if __name__ == "__main__":
    style = Image.open("egon.jpeg")
    content = Image.open("bruges.jpeg")
    style_2 = Image.open("starry.jpeg")

    onestyle = NST(content, style)
    twostyles = NST(content, style, style_2=style_2)
    onestyle.save("test.png")
    twostyles.save("test2.png")

