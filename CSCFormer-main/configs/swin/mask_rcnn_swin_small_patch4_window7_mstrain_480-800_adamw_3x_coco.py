# _base_ = [
#     '../_base_/models/mask_rcnn_swin_fpn.py',
#     '../_base_/datasets/coco_instance.py',
#     '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
# ]
_base_ = [
    '../_base_/models/mask_rcnn_swin_fpn.py',
    '../_base_/datasets/coco_detection.py', #做目标检测，修改为coco_detection.py
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]


model = dict(
    backbone=dict(
        # embed_dim=96,
        embed_dim=128, #zty修改为Base版本
        # embed_dim=192, #zty修改为L版本

        depths=[2, 2, 18, 2],

        # num_heads=[3, 6, 12, 24],
        num_heads=[4, 8, 16, 32],  #zty修改为Base版本
        # num_heads=[6, 12, 24, 48],  # zty修改为L版本

        window_size=7,
        ape=False,
        drop_path_rate=0.2,
        patch_norm=True,
        use_checkpoint=False
    ),
    # neck=dict(in_channels=[96, 192, 384, 768]))
    neck = dict(in_channels=[128, 256, 512, 1024]))  #zty修改为Base版本
    # neck = dict(in_channels=[192, 384, 768, 1536]))  # zty修改为L版本


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),

    # ZTYxiugai
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    # dict(type='LoadAnnotations', with_bbox=True, with_mask=True),


    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AutoAugment',
         policies=[
             [
                 dict(type='Resize',
                      # img_scale=[(480, 480)],
                      # img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                      #            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                      #            (736, 1333), (768, 1333), (800, 1333)],
                      img_scale=[(768, 768)],  # ZTY
                      multiscale_mode='value',

                      # keep_ratio=True)
                      keep_ratio=False)  # ZTY:True改为False
             ],
             [
                 dict(type='Resize',
                      # img_scale=[(480, 480)],
                      # img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                      img_scale=[(768, 768)],  # ZTY


                      multiscale_mode='value',

                      # keep_ratio=True),  # True改为False
                      keep_ratio=False),  # True改为False


                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(384, 600),
                      allow_negative_crop=True),
                 dict(type='Resize',
                      # img_scale=[(480, 480)],
                      # img_scale=[(480, 1333), (512, 1333), (544, 1333),
                      #            (576, 1333), (608, 1333), (640, 1333),
                      #            (672, 1333), (704, 1333), (736, 1333),
                      #            (768, 1333), (800, 1333)],
                      img_scale=[(768, 768)],  # ZTY

                      multiscale_mode='value',
                      override=True,

                      # keep_ratio=True)
                      keep_ratio=False),  # zty:True改为False

             ]
         ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),

    # ZTYxiugai
    # dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
data = dict(train=dict(pipeline=train_pipeline))

# zty修改了lr：0.0001变为0.0000125
optimizer = dict(_delete_=True, type='AdamW', lr=0.0000125, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(step=[27, 33])
# runner = dict(type='EpochBasedRunnerAmp', max_epochs=36)
runner = dict(type='EpochBasedRunner', max_epochs=500)

# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
