

def get_model(args, num_classes, returned_layers=None):
    if args.mode == 'eval' or args.mode == 'resume':
        model = torch.load(os.path.join(args.checkpoint_path, args.checkpoint_file))
        model.eval()
        return model

    if args.pretrained_hub == 1:
        backbone = torch.hub.load("facebookresearch/swav", "resnet50")
        # backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        print("Load pretrained swav backbone from Facebook hub")
    else:
        backbone = load_pretrained_swav(args.swav_file)
        backbone.padding = nn.Identity() #TODO: need to double check this
        backbone.projection_head = nn.Identity()
        backbone.prototypes = nn.Identity()
        print("Load pretrained swav backbone from OUR checkpoint")
    backbone.avgpool = nn.Identity()
    backbone.fc = nn.Identity()
    # --------------------Map resnet backbone---------------------
    # TODO: only for resnet50
    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    if min(returned_layers) <= 0 or max(returned_layers) >= 5:
        raise ValueError(f"Each returned layer should be in the range [1,4]. Got {returned_layers}")
    return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}

    new_back_bone = my_nn.IntermediateLayerGetter(backbone, return_layers=return_layers)


    # --------------------standard demo thing---------------------

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)

    # Add normalize layer based on labeled dataset
    min_size, max_size = 800, 1333
    image_mean, image_std = [0.4697, 0.4517, 0.3954], [0.2305, 0.2258, 0.2261]
    norm_layer = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
    model.transform = norm_layer

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # model.roi_heads.box_predictor.cls_score = nn.Linear(in_features=in_features, out_features=num_classes)
    # model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features=in_features, out_features=num_classes * 4)

    # ---------------------------------------------------------
    # Change backbone
    # ---------------------------------------------------------
    # model.backbone.body.load_state_dict(new_back_bone.state_dict())
    model.backbone.body = new_back_bone
    replace_bn(model, "model")

    for m in model.parameters():
        m.requires_grad = True
    
    # ---------------------------------------------------------
    # Change fpn
    # ---------------------------------------------------------
    # ResNet-18
    if args.arch == "resnet18":
        model.backbone.fpn.inner_blocks[0] = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
        model.backbone.fpn.inner_blocks[1] = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
        model.backbone.fpn.inner_blocks[2] = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        model.backbone.fpn.inner_blocks[3] = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))

    # ---------------------------------------------------------
    # Change box_head
    # ---------------------------------------------------------
    out_channels = model.backbone.out_channels
    resolution = model.roi_heads.box_roi_pool.output_size[0]
    representation_size = 1024
    new_box_head = my_nn.CustomizedBoxHead(out_channels * resolution ** 2, representation_size)
    model.roi_heads.box_head = new_box_head


    if args.debug == 1:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f'TRAIN: {name}')
            else:
                print(f'FROZEN: {name}')

        for name, child in model.named_children():
            print(f'name is: {name}')
            print(f'module is: {child}')
        # print(model.backbone.body.layer4[0].bn1.running_var)
        # print(model.backbone.body.layer4[0].bn1.running_mean)
        print("DONE")
        sys.stdout.flush()
    return model