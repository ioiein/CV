import segmentation_models_pytorch as smp


def get_model():
    # model = smp.Unet(
    #     encoder_name="efficientnet-b4",
    #     encoder_weights="imagenet",
    #     in_channels=3,
    #     classes=1
    # )
    ENCODER = 'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['car']
    ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multicalss segmentation
    DEVICE = 'cuda'
    model = smp.FPN(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=1,
        activation=ACTIVATION,
    )
    return model
