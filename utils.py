"""
Utilities

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import torch
import pickle
import numpy as np
import scipy.io as sio

from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset

from vcoco.vcoco import VCOCO
from hicodet.hicodet import HICODet

import pocket
from pocket.core import DistributedLearningEngine
from pocket.utils import DetectionAPMeter, BoxPairAssociation

import sys
# sys.path.append('detr')
import detr.datasets.transforms as T

def custom_collate(batch):
    images = []
    targets = []
    for im, tar in batch:
        images.append(im)
        targets.append(tar)
    return images, targets

class DataFactory(Dataset):
    def __init__(self, name, partition, data_root):
        if name not in ['hicodet', 'vcoco']:
            raise ValueError("Unknown dataset ", name)

        if name == 'hicodet':
            assert partition in ['train2015', 'test2015'], \
                "Unknown HICO-DET partition " + partition
            self.dataset = HICODet(
                root=os.path.join(data_root, 'hico_20160224_det/images', partition),
                anno_file=os.path.join(data_root, 'instances_{}.json'.format(partition)),
                target_transform=pocket.ops.ToTensor(input_format='dict')
            )
        else:
            assert partition in ['train', 'val', 'trainval', 'test'], \
                "Unknown V-COCO partition " + partition
            image_dir = dict(
                train='mscoco2014/train2014',
                val='mscoco2014/train2014',
                trainval='mscoco2014/train2014',
                test='mscoco2014/val2014'
            )
            self.dataset = VCOCO(
                root=os.path.join(data_root, image_dir[partition]),
                anno_file=os.path.join(data_root, 'instances_vcoco_{}.json'.format(partition)
                ), target_transform=pocket.ops.ToTensor(input_format='dict')
            )

        # Prepare dataset transforms
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        if partition.startswith('train'):
            self.transforms = T.Compose([
                T.RandomHorizontalFlip(),
                T.ColorJitter(.4, .4, .4),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        T.RandomResize(scales, max_size=1333),
                    ])
                ), normalize,
        ])
        else:
            self.transforms = T.Compose([
                T.RandomResize([800], max_size=1333),
                normalize,
            ])

        self.name = name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        image, target = self.dataset[i]
        if self.name == 'hicodet':
            target['labels'] = target['verb']
            # Convert ground truth boxes to zero-based index and the
            # representation from pixel indices to coordinates
            target['boxes_h'][:, :2] -= 1
            target['boxes_o'][:, :2] -= 1
        else:
            target['labels'] = target['actions']
            target['object'] = target.pop('objects')

        image, target = self.transforms(image, target)

        return image, target

class CacheTemplate(defaultdict):
    """A template for VCOCO cached results """
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            self[k] = v
    def __missing__(self, k):
        seg = k.split('_')
        # Assign zero score to missing actions
        if seg[-1] == 'agent':
            return 0.
        # Assign zero score and a tiny box to missing <action,role> pairs
        else:
            return [0., 0., .1, .1, 0.]

class CustomisedDLE(DistributedLearningEngine):
    def __init__(self, net, train_dataloader, test_dataloader, ood_dataloader, max_norm=0, num_classes=117, **kwargs):
        super().__init__(net, None, train_dataloader, **kwargs)
        self.max_norm = max_norm
        self.num_classes = num_classes
        self.test_dataloader = test_dataloader
        self.ood_dataloader = ood_dataloader

    def _on_each_iteration(self):
        loss_dict = self._state.net(
            *self._state.inputs, targets=self._state.targets)
        if loss_dict['interaction_loss'].isnan():
            raise ValueError(f"The HOI loss is NaN for rank {self._rank}")

        self._state.loss = sum(loss for loss in loss_dict.values())
        self._state.optimizer.zero_grad(set_to_none=True)
        self._state.loss.backward()
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self._state.net.parameters(), self.max_norm)
        self._state.optimizer.step()

    @torch.no_grad()
    def test_hico(self):
        dataloader = self.test_dataloader
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        associate = BoxPairAssociation(min_iou=0.5)
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))

        meter = DetectionAPMeter(
            600, nproc=1,
            num_gt=dataset.anno_interaction,
            algorithm='11P'
        )

        all_label = []
        all_logit = []
        for batch in tqdm(dataloader):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            output = net(inputs)

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            target = batch[-1][0]
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            objects = output['objects']
            scores = output['scores']
            verbs = output['labels']
            interactions = conversion[objects, verbs]
            # Recover target box scale
            gt_bx_h = net.module.recover_boxes(target['boxes_h'], target['size'])
            gt_bx_o = net.module.recover_boxes(target['boxes_o'], target['size'])

            # -------------- OOD 任务 ------------ #
            # ctw = output["ctw"]
            # atd = output["atd"]
            # all_ctw.append(ctw)
            # all_atd.append(atd)
            all_scores = output['all_scores']   # [ho_pairs_cnt, 117]
            ood_boxes_h, ood_boxes_o = boxes[output['all_pairings']].unbind(0)
            all_objects = output["all_objects"]

            # 仅使用匹配的边界框计算OOD性能
            # 匹配边界框，得到 ground-truth 标签(1 表示 ID 人物对，0 表示 OOD 人物对)
            # target['object']
            # all_objects
            unique_object = all_objects.unique()
            for obj_idx in unique_object:
                gt_idx = torch.nonzero(target['object'] == obj_idx).squeeze(1)
                det_idx = torch.nonzero(all_objects == obj_idx).squeeze(1)
                if len(gt_idx):
                    ood_label = associate(
                        (gt_bx_h[gt_idx].view(-1, 4),
                        gt_bx_o[gt_idx].view(-1, 4)),
                        (ood_boxes_h[det_idx].view(-1, 4),
                        ood_boxes_o[det_idx].view(-1, 4)),
                        None   # 对于重复匹配的人物对，仅保留 IoU 最大的人物对
                    )
                    # 仅保留与 ground-truth 匹配的 人物对
                    d2idxs = torch.nonzero(ood_label, as_tuple=False)
                    idxs = det_idx[d2idxs]
                    pos_score = all_scores[idxs].squeeze(1)

                    # 匹配的人物对
                    all_label.append(torch.ones_like(idxs))
                    all_logit.append(pos_score)

            # all_label.append(ood_label)
            # assert len(ood_label) == len(cur_ing_logits)
            # ---------------- END -------------- #  

            # Associate detected pairs with ground truth pairs
            labels = torch.zeros_like(scores)
            unique_hoi = interactions.unique()
            for hoi_idx in unique_hoi:
                gt_idx = torch.nonzero(target['hoi'] == hoi_idx).squeeze(1)
                det_idx = torch.nonzero(interactions == hoi_idx).squeeze(1)
                if len(gt_idx):
                    labels[det_idx] = associate(
                        (gt_bx_h[gt_idx].view(-1, 4),
                        gt_bx_o[gt_idx].view(-1, 4)),
                        (boxes_h[det_idx].view(-1, 4),
                        boxes_o[det_idx].view(-1, 4)),
                        scores[det_idx].view(-1)
                    )

            meter.append(scores, interactions, labels)

        match_ood_results = {
            "label": torch.cat(all_label).squeeze(-1).numpy(),
            "logit": torch.cat(all_logit).numpy()
        }

        return meter.eval(), match_ood_results


    @torch.no_grad()
    def test_hico_ood(self):
        dataloader = self.ood_dataloader
        net = self._state.net
        net.eval()

        associate = BoxPairAssociation(min_iou=0.5)

        all_label = []
        all_logit = []
        for batch in tqdm(dataloader):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            output = net(inputs)

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            target = batch[-1][0]
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            objects = output['objects']
            scores = output['scores']
            verbs = output['labels']
            # interactions = conversion[objects, verbs]
            # Recover target box scale
            gt_bx_h = net.module.recover_boxes(target['boxes_h'], target['size'])
            gt_bx_o = net.module.recover_boxes(target['boxes_o'], target['size'])

            # -------------- OOD 任务 ------------ #
            # ctw = output["ctw"]
            # atd = output["atd"]
            # all_ctw.append(ctw)
            # all_atd.append(atd)
            all_scores = output['all_scores']   # [ho_pairs_cnt, 117]
            ood_boxes_h, ood_boxes_o = boxes[output['all_pairings']].unbind(0)
            all_objects = output["all_objects"]

            # 仅使用匹配的边界框计算OOD性能
            # 匹配边界框，得到 ground-truth 标签(1 表示 ID 人物对，0 表示 OOD 人物对)
            # target['object']
            # all_objects
            unique_object = all_objects.unique()
            for obj_idx in unique_object:
                gt_idx = torch.nonzero(target['object'] == obj_idx).squeeze(1)
                det_idx = torch.nonzero(all_objects == obj_idx).squeeze(1)
                if len(gt_idx):
                    ood_label = associate(
                        (gt_bx_h[gt_idx].view(-1, 4),
                        gt_bx_o[gt_idx].view(-1, 4)),
                        (ood_boxes_h[det_idx].view(-1, 4),
                        ood_boxes_o[det_idx].view(-1, 4)),
                        None   # 对于重复匹配的人物对，仅保留 IoU 最大的人物对
                    )
                    # 仅保留与 ground-truth 匹配的 人物对
                    d2idxs = torch.nonzero(ood_label, as_tuple=False)
                    idxs = det_idx[d2idxs]
                    pos_score = all_scores[idxs].squeeze(1)

                    # 匹配的人物对
                    all_label.append(torch.zeros_like(idxs))
                    all_logit.append(pos_score)
            # ---------------- END -------------- #

        match_ood_results = {
            "label": torch.cat(all_label).squeeze(-1).numpy(),
            "logit": torch.cat(all_logit).numpy()
        }

        return match_ood_results

    @torch.no_grad()
    def cache_hico(self, dataloader, cache_dir='matlab'):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))
        object2int = dataset.object_to_interaction

        # Include empty images when counting
        nimages = len(dataset.annotations)
        all_results = np.empty((600, nimages), dtype=object)

        for i, batch in enumerate(tqdm(dataloader)):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            output = net(inputs)

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            # NOTE Index i is the intra-index amongst images excluding those
            # without ground truth box pairs
            image_idx = dataset._idx[i]
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            objects = output['objects']
            scores = output['scores']
            verbs = output['labels']
            interactions = conversion[objects, verbs]
            # Rescale the boxes to original image size
            ow, oh = dataset.image_size(i)
            h, w = output['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, ow / w, oh / h
            ]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct

            # Convert box representation to pixel indices
            boxes_h[:, 2:] -= 1
            boxes_o[:, 2:] -= 1

            # Group box pairs with the same predicted class
            permutation = interactions.argsort()
            boxes_h = boxes_h[permutation]
            boxes_o = boxes_o[permutation]
            interactions = interactions[permutation]
            scores = scores[permutation]

            # Store results
            unique_class, counts = interactions.unique(return_counts=True)
            n = 0
            for cls_id, cls_num in zip(unique_class, counts):
                all_results[cls_id.long(), image_idx] = torch.cat([
                    boxes_h[n: n + cls_num],
                    boxes_o[n: n + cls_num],
                    scores[n: n + cls_num, None]
                ], dim=1).numpy()
                n += cls_num
        
        # Replace None with size (0,0) arrays
        for i in range(600):
            for j in range(nimages):
                if all_results[i, j] is None:
                    all_results[i, j] = np.zeros((0, 0))
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        # Cache results
        for object_idx in range(80):
            interaction_idx = object2int[object_idx]
            sio.savemat(
                os.path.join(cache_dir, f'detections_{(object_idx + 1):02d}.mat'),
                dict(all_boxes=all_results[interaction_idx])
            )

    @torch.no_grad()
    def cache_vcoco(self, dataloader, cache_dir='vcoco_cache'):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        all_results = []
        for i, batch in enumerate(tqdm(dataloader)):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            output = net(inputs)

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            # NOTE Index i is the intra-index amongst images excluding those
            # without ground truth box pairs
            image_id = dataset.image_id(i)
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            scores = output['scores']
            actions = output['labels']
            # Rescale the boxes to original image size
            ow, oh = dataset.image_size(i)
            h, w = output['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, ow / w, oh / h
            ]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct

            for bh, bo, s, a in zip(boxes_h, boxes_o, scores, actions):
                a_name = dataset.actions[a].split()
                result = CacheTemplate(image_id=image_id, person_box=bh.tolist())
                result[a_name[0] + '_agent'] = s.item()
                result['_'.join(a_name)] = bo.tolist() + [s.item()]
                all_results.append(result)

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        with open(os.path.join(cache_dir, 'cache.pkl'), 'wb') as f:
            # Use protocol 2 for compatibility with Python2
            pickle.dump(all_results, f, 2)

def _cal_auc_fpr(id_ness, labels):
    auroc = metrics.roc_auc_score(labels, id_ness)
    fpr,tpr,thresh = Roc(labels, id_ness, pos_label=1)
    fpr = float(interpolate.interp1d(tpr, fpr)(0.95))
    return auroc, fpr

to_np = lambda x: x.detach().cpu().numpy()
def max_logit_score(logits):
    return to_np(torch.max(logits, -1)[0])
def msp_score(logits):
    prob = torch.softmax(logits, -1)
    return to_np(torch.max(prob, -1)[0])
def energy_score(logits):
    return to_np(torch.logsumexp(logits, -1))

def merge_ood_results(ood_results_lh, ood_results_rh):
    """合并两个 OOD 任务输出的结果"""
    all_results = {}
    assert ood_results_lh.keys() == ood_results_rh.keys()
    for key in ood_results_lh.keys():
        lh_res = ood_results_lh[key]
        rh_res = ood_results_rh[key]
        all_results[key] = np.concatenate((lh_res, rh_res), axis=0)
    return all_results

def evaluate_ood_results(ood_results):
    """评估 OOD 任务输出的结果"""
    # ground-truth
    label_key_name = "label"
    labels = ood_results[label_key_name]  # [n, 1]

    eval_results = {}
    for key, value in ood_results.items():
        if key == label_key_name:
            continue
        auroc, fpr = _cal_auc_fpr(id_ness=value, labels=labels)
        eval_results[key] = (auroc, fpr)
    
    return eval_results
