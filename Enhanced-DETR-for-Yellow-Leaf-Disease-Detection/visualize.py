from util.plot_utils import plot_logs, plot_precision_recall, plot_mAP
from pathlib import Path
import matplotlib.pyplot as plt
import os

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
'''
################# UNIQUE ################
train_lr
train_grad_norm
test_coco_eval_bbox
epoch
n_parameters

################# OTHER #################
class_error
loss
loss_ce
loss_bbox
loss_giou
loss_ce_0
loss_bbox_0
loss_giou_0
loss_ce_1
loss_bbox_1
loss_giou_1
loss_ce_2
loss_bbox_2
loss_giou_2
loss_ce_3
loss_bbox_3
loss_giou_3
loss_ce_4
loss_bbox_4
loss_giou_4
loss_ce_unscaled
class_error_unscaled
loss_bbox_unscaled
loss_giou_unscaled
cardinality_error_unscaled
loss_ce_0_unscaled
loss_bbox_0_unscaled
loss_giou_0_unscaled
cardinality_error_0_unscaled
loss_ce_1_unscaled
loss_bbox_1_unscaled
loss_giou_1_unscaled
cardinality_error_1_unscaled
loss_ce_2_unscaled
loss_bbox_2_unscaled
loss_giou_2_unscaled
cardinality_error_2_unscaled
loss_ce_3_unscaled
loss_bbox_3_unscaled
loss_giou_3_unscaled
cardinality_error_3_unscaled
loss_ce_4_unscaled
loss_bbox_4_unscaled
loss_giou_4_unscaled
cardinality_error_4_unscaled


'''


def get_file_path(dir_names):
    list_path = []
    list_alias = []
    for dir in dir_names.keys():
        path = os.path.join(".", "exps", dir)
        list_path.append(Path(path))
        list_alias.append(dir_names[dir])
    return list_path, list_alias


dict_path = {"original": "res50-ddetr-ss-paper",
             "resnet_deformable_detr_lr1e-4_b2": "resnet-50_ddetr",
             "mobilenet_v3_deformable_detr_b2": "mb-v3L_ddetr",
             "effi_v2s_deformable_detr": "effi-v2S_ddetr",
             "swin_deformable_detr": "swin-T_ddetr",
             "swin_ddetr_fixed": "swin-T_ddetr_fixed"
             }

# dict_path = {"original": "res50-ddetr-ss-paper",
#              "resnet_deformable_detr_lr1e-4_b2": "resnet-50_ddetr",
#              }


num_epoch = 100
list_alias=['EFF','MOB','RES']
# import pandas as pd
list_path=['D:\\DEEP\\deformable_detr_efficientnet\\exps\\effi-v2S_ddetr\\log.txt','D:/DEEP/deformable_detr_efficientnet/exps/mb-v3L_ddetr/log.txt','D:/DEEP/deformable_detr_efficientnet/exps/res-50_ddetr/log.txt']
# dfs=[]




# fields = ("loss", "class_error", "loss_bbox", "loss_giou")
# fig, _ = plot_logs(logs=list_path, fields=fields,
#                    num_epoch=num_epoch, alias=list_alias)
# fig.savefig("result_loss_per_epoch.png")


fig, _ = plot_mAP(logs=list_path, num_epoch=num_epoch, alias=list_alias)
plt.show()
fig.savefig("result_ap_per_epoch.png")
