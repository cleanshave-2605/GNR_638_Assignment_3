from models import iresnet

def build_model(args):
    model = None  # we made changes to train only iresnet and skip the other models

    if args.arch == 'iresnet':
        # Ensure depth is one of the supported iResNet depths
        if args.model_depth == 18:
            model = iresnet.iresnet18(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 34:
            model = iresnet.iresnet34(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 50:
            model = iresnet.iresnet50(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 101:
            model = iresnet.iresnet101(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 152:
            model = iresnet.iresnet152(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 200:
            model = iresnet.iresnet200(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        # Add other depths (302, 404, 1001) here if needed...

    # Safety check: If model is still None, something went wrong with arguments
    if model is None:
        raise ValueError(
            f"Could not build model. Check your arguments: "
            f"arch='{args.arch}', model_depth={args.model_depth}. "
            f"Ensure these match the logic in build_model.py"
        )

    return model
