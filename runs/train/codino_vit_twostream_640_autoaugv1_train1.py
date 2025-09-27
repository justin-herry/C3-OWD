classes = ('person', 'car', 'bicycle')
num_classes = 3
max_epochs = 12
image_size = (640, 640)
window_block_indexes = [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26
]
residual_block_indexes = []
norm_cfg = dict(type='BN', requires_grad=True)
num_dec_layer = 6
lambda_2 = 2.0
model = dict(
    type='TwoStreamCoDETR',
    train_stage=1,
    classes=('person', 'car', 'bicycle'),
    backbone=dict(
        type='ResNet',
        depth=50,
        base_channels=32,
        num_stages=4,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[256, 512, 1024],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=12.0),
        loss_bbox=dict(type='L1Loss', loss_weight=12.0)),
    query_head=dict(
        type='CoDeformDETRHead',
        num_query=300,
        num_classes=3,
        in_channels=2048,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=True,
        mixed_selection=True,
        transformer=dict(
            type='CoDeformableDetrTransformer',
            num_co_heads=2,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        dropout=0.0),
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='CoDeformableDetrTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                look_forward_twice=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.0),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            dropout=0.0)
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    roi_head=[
        dict(
            type='CoStandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[8, 16, 32, 64],
                finest_scale=112),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=3,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=12.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=120.0)))
    ],
    bbox_head=[
        dict(
            type='CoATSSHead',
            num_classes=3,
            in_channels=256,
            stacked_convs=1,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=12.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=24.0),
            loss_centerness=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=12.0))
    ],
    train_cfg=[
        dict(
            assigner=dict(
                type='HungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(
                    type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
        dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms_pre=4000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)),
        dict(
            assigner=dict(type='ATSSAssigner', topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)
    ],
    test_cfg=[
        dict(max_per_img=1000, nms=dict(type='soft_nms', iou_threshold=0.8)),
        dict(
            rpn=dict(
                nms_pre=8000,
                max_per_img=2000,
                nms=dict(type='nms', iou_threshold=0.9),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.0,
                mask_thr_binary=0.5,
                nms=dict(type='soft_nms', iou_threshold=0.5),
                max_per_img=1000)),
        dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='soft_nms', iou_threshold=0.6),
            max_per_img=500)
    ])
dataset_type = 'CocoDataset'
data_root = '/root/autodl-tmp/Deformable-DETR/data/FLIR/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline_no_copypaste = [
    dict(type='LoadPairedImageFromFile', rgb_folder_name='rgb'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PairedImagesResize',
        img_scale=(640, 640),
        ratio_range=(0.1, 2.5),
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='PairedImagesRandomCrop',
        crop_type='absolute_range',
        crop_size=(640, 640),
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(0.01, 0.01)),
    dict(type='PairedImagesRandomFlip', flip_ratio=0.5),
    dict(type='PairedImagesAutoAugmentCustom', autoaug_type='v1'),
    dict(
        type='PairedImagesPad',
        size=(640, 640),
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='PairedImagesNormalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='PairedImagesDefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'img_lwir', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadPairedImageFromFile', rgb_folder_name='rgb'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(
                type='PairedImagesResize',
                img_scale=(640, 640),
                keep_ratio=True),
            dict(type='PairedImagesRandomFlip', test_flip=False),
            dict(
                type='PairedImagesPad',
                size=(640, 640),
                pad_val=dict(img=(114, 114, 114))),
            dict(
                type='PairedImagesNormalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img', 'img_lwir']),
            dict(type='Collect', keys=['img', 'img_lwir'])
        ])
]
val_pipeline = [
    dict(type='LoadPairedImageFromFile', rgb_folder_name='rgb'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(
                type='PairedImagesResize',
                img_scale=(640, 640),
                keep_ratio=True),
            dict(type='PairedImagesRandomFlip', test_flip=False),
            dict(
                type='PairedImagesPad',
                size=(640, 640),
                pad_val=dict(img=(114, 114, 114))),
            dict(
                type='PairedImagesNormalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img', 'img_lwir']),
            dict(type='Collect', keys=['img', 'img_lwir'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='ClassBalancedDataset',
        oversample_thr=0.2,
        dataset=[
            dict(
                type='CocoDataset',
                classes=('person', 'car', 'bicycle'),
                ann_file=
                '/root/autodl-tmp/Deformable-DETR/data/FLIR/train/train.json',
                img_prefix=
                '/root/autodl-tmp/Deformable-DETR/data/FLIR/train/rgb',
                filter_empty_gt=False,
                pipeline=[
                    dict(
                        type='LoadPairedImageFromFile', rgb_folder_name='rgb'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        type='PairedImagesResize',
                        img_scale=(640, 640),
                        ratio_range=(0.1, 2.5),
                        multiscale_mode='range',
                        keep_ratio=True),
                    dict(
                        type='PairedImagesRandomCrop',
                        crop_type='absolute_range',
                        crop_size=(640, 640),
                        recompute_bbox=True,
                        allow_negative_crop=True),
                    dict(
                        type='FilterAnnotations', min_gt_bbox_wh=(0.01, 0.01)),
                    dict(type='PairedImagesRandomFlip', flip_ratio=0.5),
                    dict(
                        type='PairedImagesAutoAugmentCustom',
                        autoaug_type='v1'),
                    dict(
                        type='PairedImagesPad',
                        size=(640, 640),
                        pad_val=dict(img=(114, 114, 114))),
                    dict(
                        type='PairedImagesNormalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='PairedImagesDefaultFormatBundle'),
                    dict(
                        type='Collect',
                        keys=['img', 'img_lwir', 'gt_bboxes', 'gt_labels'])
                ])
        ]),
    val=dict(
        type='CocoDataset',
        classes=('person', 'car', 'bicycle'),
        ann_file='/root/autodl-tmp/Deformable-DETR/data/FLIR/val/val.json',
        img_prefix='/root/autodl-tmp/Deformable-DETR/data/FLIR/val/rgb/',
        pipeline=[
            dict(type='LoadPairedImageFromFile', rgb_folder_name='rgb'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(
                        type='PairedImagesResize',
                        img_scale=(640, 640),
                        keep_ratio=True),
                    dict(type='PairedImagesRandomFlip', test_flip=False),
                    dict(
                        type='PairedImagesPad',
                        size=(640, 640),
                        pad_val=dict(img=(114, 114, 114))),
                    dict(
                        type='PairedImagesNormalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img', 'img_lwir']),
                    dict(type='Collect', keys=['img', 'img_lwir'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        classes=('person', 'car', 'bicycle'),
        ann_file='/root/autodl-tmp/Deformable-DETR/data/FLIR/val/val.json',
        img_prefix='/root/autodl-tmp/Deformable-DETR/data/FLIR/val/rgb/',
        pipeline=[
            dict(type='LoadPairedImageFromFile', rgb_folder_name='rgb'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(
                        type='PairedImagesResize',
                        img_scale=(640, 640),
                        keep_ratio=True),
                    dict(type='PairedImagesRandomFlip', test_flip=False),
                    dict(
                        type='PairedImagesPad',
                        size=(640, 640),
                        pad_val=dict(img=(114, 114, 114))),
                    dict(
                        type='PairedImagesNormalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img', 'img_lwir']),
                    dict(type='Collect', keys=['img', 'img_lwir'])
                ])
        ]))
evaluation = dict(save_best='auto', interval=1, metric=['bbox'])
dist_params = dict(backend='nccl')
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
steps = [8, 11]
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=250,
    warmup_ratio=0.01,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
optimizer = dict(
    type='AdamW',
    lr=0.001,
    weight_decay=0.05,
    paramwise_cfg=dict(custom_keys=dict(backbone=dict(lr_mult=0.1))))
checkpoint_config = dict(by_epoch=True, interval=1, max_keep_ckpts=1)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
log_level = 'INFO'
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
load_from = None
resume_from = None
work_dir = 'runs/train/'
auto_resume = False
gpu_ids = [0]
