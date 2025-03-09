# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
import torch.nn as nn


@dataclass
class AdjustLabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    ignore_eos: bool = field(
        default=False,
        metadata={"help": "Ignore eos token"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    drop_worst_ratio: float = field(
        default=0.0,
        metadata={"help": "ratio for discarding bad samples"},
    )
    drop_worst_after: int = field(
        default=0,
        metadata={"help": "steps for discarding bad samples"},
    )
    use_rdrop: bool = field(
        default=False, metadata={"help": "use R-Drop"}
    )
    reg_alpha: float = field(
        default=1.0, metadata={"help": "weight for R-Drop"}
    )
    sample_patch_num: int = field(
        default=196, metadata={"help": "sample patches for v1"}
    )
    constraint_range: Optional[str] = field(
        default=None,
        metadata={"help": "constraint range"}
    )
    disease_weight: float = field(
        default=1,
        metadata={"help": "Weight for disease loss"},
    )
    location_weight: float = field(
        default=1,
        metadata={"help": "Weight for location loss"},
    )
    severity_weight: float = field(
        default=1,
        metadata={"help": "Weight for severity loss"},
    )
    gamma: float = field(
        default=1.5,
        metadata={"help": "Exponent of the focal loss, a bigger value reduces the contribution of easy samples."},
    )


def construct_rdrop_sample(x):
    if isinstance(x, dict):
        for key in x:
            x[key] = construct_rdrop_sample(x[key])
        return x
    elif isinstance(x, torch.Tensor):
        return x.repeat(2, *([1] * (x.dim()-1)))
    elif isinstance(x, int):
        return x * 2
    elif isinstance(x, np.ndarray):
        return x.repeat(2)
    else:
        raise NotImplementedError


def kl_loss(p, q):
    p_loss = F.kl_div(p, torch.exp(q), reduction='sum')
    q_loss = F.kl_div(q, torch.exp(p), reduction='sum')
    loss = (p_loss + q_loss) / 2
    return loss


def label_smoothed_nll_loss(
        lprobs, target, epsilon, update_num, reduce=True,
        drop_worst_ratio=0.0, drop_worst_after=0, use_rdrop=False, reg_alpha=1.0,
        constraint_masks=None, constraint_start=None, constraint_end=None
):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target).squeeze(-1)
    if constraint_masks is not None:
        smooth_loss = -lprobs.masked_fill(~constraint_masks, 0).sum(dim=-1, keepdim=True).squeeze(-1)
        eps_i = epsilon / (constraint_masks.sum(1) - 1 + 1e-6)
    elif constraint_start is not None and constraint_end is not None:
        constraint_range = [0, 1, 2, 3] + list(range(constraint_start, constraint_end))
        smooth_loss = -lprobs[:, constraint_range].sum(dim=-1, keepdim=True).squeeze(-1)
        eps_i = epsilon / (len(constraint_range) - 1 + 1e-6)
    else:
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True).squeeze(-1)
        eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    if drop_worst_ratio > 0 and update_num > drop_worst_after:
        if use_rdrop:
            true_batch_size = loss.size(0) // 2
            _, indices = torch.topk(loss[:true_batch_size], k=int(true_batch_size * (1 - drop_worst_ratio)), largest=False)
            loss = torch.cat([loss[indices], loss[indices+true_batch_size]])
            nll_loss = torch.cat([nll_loss[indices], nll_loss[indices+true_batch_size]])
            lprobs = torch.cat([lprobs[indices], lprobs[indices+true_batch_size]])
        else:
            loss, indices = torch.topk(loss, k=int(loss.shape[0] * (1 - drop_worst_ratio)), largest=False)
            nll_loss = nll_loss[indices]
            lprobs = lprobs[indices]

    ntokens = loss.numel()
    nll_loss = nll_loss.sum()
    loss = loss.sum()
    if use_rdrop:
        true_batch_size = lprobs.size(0) // 2
        p = lprobs[:true_batch_size]
        q = lprobs[true_batch_size:]
        if constraint_start is not None and constraint_end is not None:
            constraint_range = [0, 1, 2, 3] + list(range(constraint_start, constraint_end))
            p = p[:, constraint_range]
            q = q[:, constraint_range]
        loss += kl_loss(p, q) * reg_alpha

    return loss, nll_loss, ntokens


@register_criterion(
    "adjust_label_smoothed_cross_entropy", dataclass=AdjustLabelSmoothedCrossEntropyCriterionConfig
)
class AdjustLabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        ignore_eos=False,
        report_accuracy=False,
        drop_worst_ratio=0,
        drop_worst_after=0,
        use_rdrop=False,
        reg_alpha=1.0,
        sample_patch_num=196,
        constraint_range=None,
        disease_weight = 1,
        location_weight = 1,
        severity_weight = 1,
        gamma = 1.5
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.ignore_eos = ignore_eos
        self.report_accuracy = report_accuracy
        self.drop_worst_ratio = drop_worst_ratio
        self.drop_worst_after = drop_worst_after
        self.use_rdrop = use_rdrop
        self.reg_alpha = reg_alpha
        self.sample_patch_num = sample_patch_num
        # Convert the scalars to tensors and register them as buffers so they follow model device moves
        self.register_buffer('disease_weight', torch.tensor(disease_weight))
        self.register_buffer('location_weight', torch.tensor(location_weight))
        self.register_buffer('severity_weight', torch.tensor(severity_weight))
        self.register_buffer('gamma', torch.tensor(gamma))
        # alpha follows normalize inverse frequency of the training data
        # additionally it has undergone a power transformation
        alpha = [
                0.36, 0.37, 0.37, 0.39, 0.39, 0.41, 0.42, 0.43, 0.44, 0.45,
                0.46, 0.46, 0.47, 0.48, 0.50, 0.51, 0.51, 0.53, 0.53, 0.54,
                0.58, 0.60, 0.60, 0.61, 0.61, 0.62, 0.68, 0.73, 0.78, 0.91, 1.00
            ]
        self.register_buffer('alpha', torch.tensor(alpha))

        self.constraint_start = None
        self.constraint_end = None
        if constraint_range is not None:
            constraint_start, constraint_end = constraint_range.split(',')
            self.constraint_start = int(constraint_start)
            self.constraint_end = int(constraint_end)

    def forward(self, model, sample, update_num=0, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if isinstance(sample, list):
            if self.sample_patch_num > 0:
                sample[0]['net_input']['sample_patch_num'] = self.sample_patch_num
            loss_v1, sample_size_v1, logging_output_v1 = self.forward(model, sample[0], update_num, reduce)
            loss_v2, sample_size_v2, logging_output_v2 = self.forward(model, sample[1], update_num, reduce)

            
            loss = loss_v1 / sample_size_v1 + loss_v2 / sample_size_v2
            
            sample_size = 1
            logging_output = {
                "loss": loss.data,
                "loss_v1": loss_v1.data,
                "loss_v2": loss_v2.data,
                "nll_loss": logging_output_v1["nll_loss"].data / sample_size_v1 + logging_output_v2["nll_loss"].data / sample_size_v2,
                "ntokens": logging_output_v1["ntokens"] + logging_output_v2["ntokens"],
                "nsentences": logging_output_v1["nsentences"] + logging_output_v2["nsentences"],
                "sample_size": 1,
                "sample_size_v1": sample_size_v1,
                "sample_size_v2": sample_size_v2,
            }
            return loss, sample_size, logging_output
        

        if self.use_rdrop:
            construct_rdrop_sample(sample)

        net_output = model(**sample["net_input"])
        loss, nll_loss, ntokens = self.compute_loss(model, net_output, sample, update_num, reduce=reduce)

        self.gamma = self.gamma.to(loss.device)
        self.disease_weight = self.disease_weight.to(loss.device)
        self.location_weight = self.location_weight.to(loss.device)
        self.severity_weight = self.disease_weight.to(loss.device)
        # Compute auxiliary BCE loss
        disease_loss, disease_samples = self.compute_disease_loss(net_output, sample)
        #location_loss, location_samples = self.compute_location_loss(net_output, sample)
        #severity_loss, severity_samples = self.compute_severity_loss(net_output, sample)
        
        aux_loss = self.disease_weight * disease_loss #+ self.location_weight * location_loss + self.severity_weight * severity_loss
        aux_sample_size = self.disease_weight * disease_samples #+ self.location_weight * location_samples + self.severity_weight * severity_samples
        
        total_loss = loss + aux_loss

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else ntokens
        )
        total_sample_size = sample_size + aux_sample_size
        logging_output = {
            "loss": total_loss.data,
            "main_loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
            "disease_samples": disease_samples,
        #    "location_samples": location_samples,
        #    "severity_samples": severity_samples,
            "disease_loss": disease_loss.data,
        #    "location_loss": location_loss.data,
        #    "severity_loss": severity_loss.data,
        }

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return total_loss, total_sample_size, logging_output

    def compute_disease_loss(self, net_output, sample):
        
        disease_logits_1 = net_output[1]["disease_logits"][0]
        disease_targets_1 = sample["net_input"]["disease_tensor_gt"]
        
        if "disease_logits_2" in net_output[1]:
            disease_logits_2 = net_output[1]["disease_logits_2"][0]
        if "disease_tensor_gt_2" in sample["net_input"]:
            disease_targets_2 = sample["net_input"]["disease_tensor_gt_2"]
        if "ref_id_mask" in sample["net_input"]:
            ref_id_mask = sample["net_input"]["ref_id_mask"]

        self.alpha = self.alpha.view(1, disease_targets_1.shape[1]).to(disease_targets_1.device)  # Shape [1, num_diseases]

        # BCEWithLogitsLoss without reduction to get per-element loss
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        # Compute per-element loss for both sets; shape: [B, num_diseases]
        loss_1 = loss_fn(disease_logits_1, disease_targets_1)
        loss_2 = loss_fn(disease_logits_2, disease_targets_2) 

        # Compute the probabilities from logits
        probs_1 = torch.sigmoid(disease_logits_1)
        probs_2 = torch.sigmoid(disease_logits_2)

        # Compute p_t: for targets=1 use the probability, for targets=0 use (1 - probability)
        p_t1 = disease_targets_1 * probs_1 + (1 - disease_targets_1) * (1 - probs_1)
        p_t2 = disease_targets_2 * probs_2 + (1 - disease_targets_2) * (1 - probs_2)

        # Compute the modulating factor: (1 - p_t)^gamma
        modulating_factor1 = (1 - p_t1) ** self.gamma
        modulating_factor2 = (1 - p_t2) ** self.gamma

        # Compute the focal loss for each element
        focal_loss_1 = self.alpha * modulating_factor1 * loss_1
        focal_loss_2 = self.alpha * modulating_factor2 * loss_2

        # Create a mask for valid samples in the second image:
        # Valid sample => ref_id_mask == -1, so we get a 1 if valid, 0 if not.
        mask_2 = (ref_id_mask == -1).float().unsqueeze(1)  # shape: [B, 1]

        # Broadcast mask_2 to [B, num_diseases] and apply it.
        focal_loss_2_masked = focal_loss_2 * mask_2
        
        # Compute total number of valid elements:
        # For loss_1, every element is valid: B 
        # For loss_2, only valid samples count
        num_valid_loss1 = focal_loss_1.shape[0]  # B 
        num_valid_loss2 = mask_2.sum()
        valid_samples = num_valid_loss1 + num_valid_loss2

        # Sum all the losses 
        focal_loss = focal_loss_1.sum() + focal_loss_2_masked.sum()

        return focal_loss, valid_samples

    def compute_location_loss(self, net_output, sample):
        """
        Compute the loss for the location classifier.
        """
        
        disease_present_1 = sample["net_input"]["disease_present"]
        location_logits_1 = net_output[1]["location_logits"][0]
        location_targets_1 = sample["net_input"]["location_tensor_gt"]

        # Reshape alpha to match broadcasting across the num_diseases dimension
        self.alpha = self.alpha.view(1, location_targets_1.shape[1], 1).to(location_targets_1.device)  # Shape [1, num_diseases, 1]
        
        if "location_logits_2" in net_output[1]:
            location_logits_2 = net_output[1]["location_logits_2"][0]
        if "location_tensor_gt_2" in sample["net_input"]:
            location_targets_2 = sample["net_input"]["location_tensor_gt_2"]
        if "ref_id_mask" in sample["net_input"]:
            ref_id_mask = sample["net_input"]["ref_id_mask"]
        if "disease_present_2" in sample["net_input"]:
            disease_present_2 = sample["net_input"]["disease_present_2"]
        
        num_locations = location_targets_1.shape[2]
        
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        
        # Compute raw per-element BCE loss.
        # Shapes: [B, num_diseases, location_dim]
        loss_1 = loss_fn(location_logits_1, location_targets_1)
        loss_2 = loss_fn(location_logits_2, location_targets_2)

        # Compute the probabilities from logits
        probs_1 = torch.sigmoid(location_logits_1)
        probs_2 = torch.sigmoid(location_logits_2)

        # Compute p_t: for targets=1 use the probability, for targets=0 use (1 - probability)
        p_t1 = location_targets_1 * probs_1 + (1 - location_targets_1) * (1 - probs_1)
        p_t2 = location_targets_2 * probs_2 + (1 - location_targets_2) * (1 - probs_2)

        # Compute the modulating factor: (1 - p_t)^gamma
        modulating_factor1 = (1 - p_t1) ** self.gamma
        modulating_factor2 = (1 - p_t2) ** self.gamma

        # Compute the focal loss for each element
        focal_loss_1 = self.alpha * modulating_factor1 * loss_1
        focal_loss_2 = self.alpha * modulating_factor2 * loss_2
        
        # Create a mask for valid disease rows.
        # We assume that if the first element of a disease row is -1, the entire row is invalid.
        # These masks have shape [B, num_diseases]
        valid_mask_1 = (location_targets_1[:, :, 0] != -1).float()
        valid_mask_2 = (location_targets_2[:, :, 0] != -1).float()
        
        # Expand the masks along the location_dim so they can be applied elementwise.
        # New shapes: [B, num_diseases, 1] (which will broadcast to location_dim)
        valid_mask_1 = valid_mask_1.unsqueeze(-1)
        valid_mask_2 = valid_mask_2.unsqueeze(-1)
        
        disease_mask_1 = (disease_present_1 == 1).float() # [B]
        disease_mask_1 = disease_mask_1.unsqueeze(1).unsqueeze(2)  # shape becomes [B, 1, 1]

        disease_mask_2 = (disease_present_2 == 1).float() # [B]
        disease_mask_2 = disease_mask_2.unsqueeze(1).unsqueeze(2)  # shape becomes [B, 1, 1]

        # For image 2, further zero out the loss for samples with ref_id_mask != -1.
        # Here we treat each sample as a single valid unit regardless of the number of diseases.
        sample_mask_2 = (ref_id_mask == -1).float().unsqueeze(1).unsqueeze(2)  # shape: [B,1,1]

        # For each element in the loss, the valid mask is:
        combined_mask_1 = valid_mask_1 * disease_mask_1  # shape [B, num_diseases, 1]
        combined_mask_2 = valid_mask_2 * disease_mask_2 * sample_mask_2  # shape [B, num_diseases, 1]
        loss_1_masked = focal_loss_1 * combined_mask_1
        loss_2_masked = focal_loss_2 * combined_mask_2

        loss = loss_1_masked.sum(dim=(0, 1, 2)) + loss_2_masked.sum(dim=(0, 1, 2))
        
        # Multiply by num_locations to count each location per valid row:
        valid_count_image1 = (combined_mask_1.sum() * num_locations)
        valid_count_image2 = (combined_mask_2.sum() * num_locations)
        
        total_valid_samples = valid_count_image1 + valid_count_image2
        
        return loss, total_valid_samples

    def compute_severity_loss(self, net_output, sample):
        """
        Compute the loss for the severity classifier.
        """

        disease_present_1 = sample["net_input"]["disease_present"]
        severity_logits_1 = net_output[1]["severity_logits"][0]
        severity_targets_1 = sample["net_input"]["severity_tensor_gt"]

        # Reshape alpha to match broadcasting across the num_diseases dimension
        self.alpha = self.alpha.view(1, severity_targets_1.shape[1], 1).to(severity_targets_1.device)  # Shape [1, num_diseases, 1]
        
        if "severity_logits_2" in net_output[1]:
            severity_logits_2 = net_output[1]["severity_logits_2"][0]
        if "severity_tensor_gt_2" in sample["net_input"]:
            severity_targets_2 = sample["net_input"]["severity_tensor_gt_2"]
        if "ref_id_mask" in sample["net_input"]:
            ref_id_mask = sample["net_input"]["ref_id_mask"]
        if "disease_present_2" in sample["net_input"]:
            disease_present_2 = sample["net_input"]["disease_present_2"]
        
        num_severity = severity_targets_1.shape[2]
        
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        
        # Compute raw per-element BCE loss.
        # Shapes: [B, num_diseases, severity_dim]
        loss_1 = loss_fn(severity_logits_1, severity_targets_1)
        loss_2 = loss_fn(severity_logits_2, severity_targets_2)

        # Compute the probabilities from logits
        probs_1 = torch.sigmoid(severity_logits_1)
        probs_2 = torch.sigmoid(severity_logits_2)

        # Compute p_t: for targets=1 use the probability, for targets=0 use (1 - probability)
        p_t1 = severity_targets_1 * probs_1 + (1 - severity_targets_1) * (1 - probs_1)
        p_t2 = severity_targets_2 * probs_2 + (1 - severity_targets_2) * (1 - probs_2)

        # Compute the modulating factor: (1 - p_t)^gamma
        modulating_factor1 = (1 - p_t1) ** self.gamma
        modulating_factor2 = (1 - p_t2) ** self.gamma

        # Compute the focal loss for each element
        focal_loss_1 = self.alpha * modulating_factor1 * loss_1
        focal_loss_2 = self.alpha * modulating_factor2 * loss_2
        
        # Create a mask for valid disease rows.
        # We assume that if the first element of a disease row is -1, the entire row is invalid.
        # These masks have shape [B, num_diseases]
        valid_mask_1 = (severity_targets_1[:, :, 0] != -1).float()
        valid_mask_2 = (severity_targets_2[:, :, 0] != -1).float()
        
        # Expand the masks along the severity_dim so they can be applied elementwise.
        # New shapes: [B, num_diseases, 1] (which will broadcast to severity_dim)
        valid_mask_1 = valid_mask_1.unsqueeze(-1)
        valid_mask_2 = valid_mask_2.unsqueeze(-1)
        
        disease_mask_1 = (disease_present_1 == 1).float() # [B]
        disease_mask_1 = disease_mask_1.unsqueeze(1).unsqueeze(2)  # shape becomes [B, 1, 1]

        disease_mask_2 = (disease_present_2 == 1).float() # [B]
        disease_mask_2 = disease_mask_2.unsqueeze(1).unsqueeze(2)  # shape becomes [B, 1, 1]

        # For image 2, further zero out the loss for samples with ref_id_mask != -1.
        # Here we treat each sample as a single valid unit regardless of the number of diseases.
        sample_mask_2 = (ref_id_mask == -1).float().unsqueeze(1).unsqueeze(2)  # shape: [B,1,1]

        # For each element in the loss, the valid mask is:
        combined_mask_1 = valid_mask_1 * disease_mask_1  # shape [B, num_diseases, 1]
        combined_mask_2 = valid_mask_2 * disease_mask_2 * sample_mask_2  # shape [B, num_diseases, 1]
        loss_1_masked = focal_loss_1 * combined_mask_1
        loss_2_masked = focal_loss_2 * combined_mask_2

        loss = loss_1_masked.sum(dim=(0, 1, 2)) + loss_2_masked.sum(dim=(0, 1, 2))
        
        # Multiply by num_severity to count each severity per valid row:
        valid_count_image1 = (combined_mask_1.sum() * num_severity)
        valid_count_image2 = (combined_mask_2.sum() * num_severity)
        
        total_valid_samples = valid_count_image1 + valid_count_image2
        
        return loss, total_valid_samples

    def get_lprobs_and_target(self, model, net_output, sample):
        conf = sample['conf'][:, None, None] if 'conf' in sample and sample['conf'] is not None else 1
        constraint_masks = None
        if "constraint_masks" in sample and sample["constraint_masks"] is not None:
            constraint_masks = sample["constraint_masks"]
            net_output[0] = net_output[0].masked_fill(~constraint_masks, -math.inf)
        if self.constraint_start is not None and self.constraint_end is not None:
            net_output[0][:, :, 4:self.constraint_start] = -math.inf
            net_output[0][:, :, self.constraint_end:] = -math.inf
        lprobs = model.get_normalized_probs(net_output, log_probs=True) * conf
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
            if constraint_masks is not None:
                constraint_masks = constraint_masks[:, self.ignore_prefix_size :, :].contiguous()
        if self.ignore_eos:
            bsz, seq_len, embed_dim = lprobs.size()
            eos_indices = target.eq(self.task.tgt_dict.eos())
            lprobs = lprobs[~eos_indices].reshape(bsz, seq_len-1, embed_dim)
            target = target[~eos_indices].reshape(bsz, seq_len-1)
            if constraint_masks is not None:
                constraint_masks = constraint_masks[~eos_indices].reshape(bsz, seq_len-1, embed_dim)
        if constraint_masks is not None:
            constraint_masks = constraint_masks.view(-1, constraint_masks.size(-1))
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1), constraint_masks

    def compute_loss(self, model, net_output, sample, update_num, reduce=True):
        lprobs, target, constraint_masks = self.get_lprobs_and_target(model, net_output, sample)
        if constraint_masks is not None:
            constraint_masks = constraint_masks[target != self.padding_idx]
        lprobs = lprobs[target != self.padding_idx]
        target = target[target != self.padding_idx]
        loss, nll_loss, ntokens = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            update_num,
            reduce=reduce,
            drop_worst_ratio=self.drop_worst_ratio,
            drop_worst_after=self.drop_worst_after,
            use_rdrop=self.use_rdrop,
            reg_alpha=self.reg_alpha,
            constraint_masks=constraint_masks,
            constraint_start=self.constraint_start,
            constraint_end=self.constraint_end
        )
        return loss, nll_loss, ntokens

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        main_loss_sum = sum(log.get("main_loss", 0) for log in logging_outputs)
        loss_sum_v1 = sum(log.get("loss_v1", 0) for log in logging_outputs)
        loss_sum_v2 = sum(log.get("loss_v2", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        sample_size_v1 = sum(log.get("sample_size_v1", 0) for log in logging_outputs)
        sample_size_v2 = sum(log.get("sample_size_v2", 0) for log in logging_outputs)
        disease_samples = sum(log.get("disease_samples", 0) for log in logging_outputs)
        #location_samples = sum(log.get("location_samples", 0) for log in logging_outputs)
        #severity_samples = sum(log.get("severity_samples", 0) for log in logging_outputs)
        disease_loss_sum = sum(log.get("disease_loss", 0) for log in logging_outputs)
        #location_loss_sum = sum(log.get("location_loss", 0) for log in logging_outputs)
        #severity_loss_sum = sum(log.get("severity_loss", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "main_loss", main_loss_sum / sample_size, sample_size, round=3
        )
        
        metrics.log_scalar(
            "disease_loss", disease_loss_sum / disease_samples, disease_samples, round=3
        )
        #metrics.log_scalar(
        #    "location_loss", location_loss_sum / location_samples, location_samples, round=3
        #)
        #metrics.log_scalar(
        #    "severity_loss", severity_loss_sum / severity_samples, severity_samples, round=3
        #)
        metrics.log_scalar(
            "loss_v1", loss_sum_v1 / max(sample_size_v1, 1), max(sample_size_v1, 1), round=3
        )
        metrics.log_scalar(
            "loss_v2", loss_sum_v2 / max(sample_size_v2, 1), max(sample_size_v2, 1), round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / sample_size, ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        metrics.log_scalar(
            "ntokens", ntokens, 1, round=3
        )
        metrics.log_scalar(
            "nsentences", nsentences, 1, round=3
        )
        metrics.log_scalar(
            "sample_size", sample_size, 1, round=3
        )
        metrics.log_scalar(
            "sample_size_v1", sample_size_v1, 1, round=3
        )
        metrics.log_scalar(
            "sample_size_v2", sample_size_v2, 1, round=3
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
