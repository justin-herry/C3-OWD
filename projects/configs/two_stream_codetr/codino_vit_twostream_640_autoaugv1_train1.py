classes = ('person','car','bicycle')   # FLIR
num_classes = len(classes)
max_epochs = 12
image_size = (640, 640)  
window_block_indexes = (
    list(range(0, 3)) + list(range(4, 7)) + list(range(8, 11)) + list(range(12, 15)) + list(range(16, 19)) +
    list(range(20, 23)) + list(range(24, 27)))   # for coco
residual_block_indexes = []
norm_cfg = dict(type='BN', requires_grad=True)
num_dec_layer = 6
lambda_2 = 2.0

model = dict(
    type='TwoStreamCoDETR',
    train_stage=1,
    classes=classes,
    moco_momentum=0.999,       
    moco_temperature=0.07,         
    feature_bank_size=4096,       
    contrastive_weight=0.5,    

    
    backbone=dict(
        type='ResNet',
        depth=50,
        base_channels=64,
        num_stages=4,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        out_indices=(0, 1, 2, ),
        frozen_stages=1,
        norm_cfg=norm_cfg,
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
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0*num_dec_layer*lambda_2),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0*num_dec_layer*lambda_2)),
    query_head=dict(
        type='CoDeformDETRHead',
        num_query=300,
        num_classes=num_classes,
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
                        type='MultiScaleDeformableAttention', embed_dims=256, dropout=0.0),
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='CoDeformableDetrTransformerDecoder',
                num_layers=num_dec_layer,
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
    roi_head=[dict(
        type='CoStandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[8, 16, 32, 64],
            finest_scale=112),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=num_classes,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            reg_decoded_bbox=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0*num_dec_layer*lambda_2),
            loss_bbox=dict(type='GIoULoss', loss_weight=10.0*num_dec_layer*lambda_2)))],
    bbox_head=[dict(
        type='CoATSSHead',
        num_classes=num_classes,
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
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0*num_dec_layer*lambda_2),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0*num_dec_layer*lambda_2),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0*num_dec_layer*lambda_2)),],
    # model training and testing settings
    train_cfg=[
        dict(
            assigner=dict(
                type='HungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
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
            debug=False),],
    test_cfg=[
        dict(
            max_per_img=1000,
            nms=dict(type='soft_nms', iou_threshold=0.8)),
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
            max_per_img=500),
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
        img_scale=image_size,
        ratio_range=(0.1, 2.5), 
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='PairedImagesRandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,     
        recompute_bbox=True,
        allow_negative_crop=True),    
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='PairedImagesRandomFlip', flip_ratio=0.5),
    dict(type='PairedImagesAutoAugmentCustom', autoaug_type='v1'), 
    dict(type='PairedImagesPad', size=image_size, pad_val=dict(img=(114, 114, 114))),  
    dict(type='PairedImagesNormalize', **img_norm_cfg),
    dict(type='PairedImagesDefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'img_lwir', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadPairedImageFromFile', rgb_folder_name='rgb'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=image_size,
        flip=False,
        transforms=[
            dict(
                type='PairedImagesResize',
                img_scale=image_size,
                keep_ratio=True),
            dict(type='PairedImagesRandomFlip', test_flip=False),
            dict(type='PairedImagesPad', size=image_size, pad_val=dict(img=(114, 114, 114))), 
            dict(type='PairedImagesNormalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img', 'img_lwir']),
            dict(type='Collect', keys=['img', 'img_lwir'])
        ])
]
val_pipeline = test_pipeline
val_pipeline[0]['rgb_folder_name'] = 'rgb'

data = dict(
    samples_per_gpu=4,   
    workers_per_gpu=2,

    train=dict(
        type='ClassBalancedDataset',
        oversample_thr=0.2,
        dataset=[
            dict(
                type=dataset_type,
                classes=classes,
                ann_file=data_root + 'train/train.json',
                img_prefix=data_root + 'train/rgb',
                filter_empty_gt=False,
                pipeline=train_pipeline_no_copypaste),
        ]
    ),

    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'val/val.json',
        img_prefix=data_root + 'val/rgb/',
        pipeline=val_pipeline
        ),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'val/val.json',
        img_prefix=data_root + 'val/rgb/',
        pipeline=val_pipeline
        ))
evaluation = dict(save_best='auto', interval=1, metric=['bbox'])

dist_params = dict(backend='nccl')

optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))

if max_epochs == 12:
    steps = [8, 11]
elif max_epochs == 20:
    steps = [16, 19]
elif max_epochs == 24:
    steps = [16, 23]
elif max_epochs == 36:
    steps = [28, 34]
else:
    assert False

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=250,
    warmup_ratio=0.01,
    step=steps)
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

# Fixed optimizer configuration - removed conflicting paramwise_cfg
optimizer = dict(
    type='AdamW',
    lr=5e-5,
    weight_decay=0.05,
    # Removed paramwise_cfg to avoid parameter group conflicts
    # If you need different learning rates for different components,
    # you'll need to ensure the parameter names match your actual model structure
)

checkpoint_config = dict(by_epoch=True, interval=1, max_keep_ckpts=1)

log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

custom_hooks = [
    dict(type='NumClassCheckHook'),
    # Removed AdaptiveLossSchedulerHook and MomentumUpdateHook as they are not in the hook registry
    # If you have custom hooks, make sure they are properly registered in your codebase
]

dist_params = dict(backend='nccl')
log_level = 'INFO'

workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'

auto_scale_lr = dict(
    enable=False, 
    base_batch_size=16,
    contrastive_lr_scale=1.0
)

load_from = '/root/autodl-tmp/Co-DETR/checkpoints/co-detr_res50_vis/co_deformable_detr_r50_1x_coco.pth'
resume_from = None

work_dir = './work_dirs/codino_vit_twostream_640_autoaugv1_train1'

train_enhancement = dict(
    adaptive_scheduling=True,
    loss_weights=dict(
        ovod_base_weight=150.0,        
        contrastive_base_weight=0.5,     
        focal_weight=2.0,            
        adaptive_factor=1.0           
    ),
    plateau_detection=dict(
        threshold=0.05,             
        patience=8,                  
        min_improvement=0.01          
    ),
    visualization=dict(
        enabled=True,                   
        interval=500,                    
        save_path='debug_visualizations', 
        max_classes_shown=8              
    ),

    feature_bank=dict(
        size=4096,                       
        temperature=0.07,               
        update_momentum=0.999,          
        negative_sampling_ratio=0.3      
    )
)

experiment_config = dict(
    name=f'enhanced_codetr_moco_{max_epochs}epochs',
    description='Enhanced TwoStreamCoDETR with MoCo contrastive learning',
    tags=['moco', 'contrastive', 'adaptive_loss', 'flir', 'two_stream'],
    notes='use moco',
    
    hyperparameters=dict(
        moco_momentum=0.999,
        moco_temperature=0.07,
        feature_bank_size=65536,
        contrastive_weight=5,
        max_epochs=max_epochs,
        base_lr=5e-5,
        batch_size=4,
        image_size=image_size
    )
)