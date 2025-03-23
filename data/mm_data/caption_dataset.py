# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from io import BytesIO

import logging
import warnings
import string

import numpy as np
import torch
import base64
from torchvision import transforms

from PIL import Image, ImageFile

from data import data_utils
from data.ofa_dataset import OFADataset

import pandas as pd
import ast 
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def collate(samples, pad_idx, eos_idx, aux=True):
    if len(samples) == 0:
        return {}

    def merge(key):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx=eos_idx,
        )

    id = np.array([s["id"] for s in samples])
    src_tokens = merge("source")
    src_lengths = torch.LongTensor([s["source"].ne(pad_idx).long().sum() for s in samples])

    patch_images = torch.stack([sample['patch_image'] for sample in samples], dim=0)
    #print('222',samples[0])
    if samples[0].get("patch_image_2", None) is not None:        
        patch_images_2 = torch.stack([sample['patch_image_2'] for sample in samples], dim=0)
    else:
        patch_images_2 = None

    patch_masks = torch.cat([sample['patch_mask'] for sample in samples])

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge("target")
        tgt_lengths = torch.LongTensor([s["target"].ne(pad_idx).long().sum() for s in samples])
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens")
    else:
        ntokens = src_lengths.sum().item()

    net_input = {
        "src_tokens": src_tokens,
        "src_lengths": src_lengths,
        "patch_images": patch_images,
        "patch_images_2": patch_images_2,
        "patch_masks": patch_masks,
        "prev_output_tokens": prev_output_tokens,
    }
    device = src_tokens.device
    
    if aux:
        # Define the auxiliary keys for the main image.
        aux_keys = [
            "disease_tensor_gt",
            "disease_present",
        ]
        for key in aux_keys:
            if key in samples[0]:
                try:
                    net_input[key] = torch.stack([s[key] for s in samples], dim=0).to(device)
                    #print(f"{key}:{net_input[key]}")
                except Exception:
                    net_input[key] = torch.tensor([s[key] for s in samples]).to(device)
        
        # For the reference (patch_image_2) auxiliary data, if available.
        if patch_images_2 is not None:
            aux_keys_2 = [
                "disease_tensor_gt_2",
                "ref_id_mask",
                "disease_present_2",
            ]
            for key in aux_keys_2:
                if key in samples[0]:
                    try:
                        net_input[key] = torch.stack([s[key] for s in samples], dim=0).to(device)
                        #print(f"{key}:{net_input[key]}")
                    except Exception:
                        net_input[key] = torch.tensor([s[key] for s in samples]).to(device)

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": net_input,
        "target": target,
    }

    return batch


class CaptionDataset(OFADataset):
    def __init__(
        self,
        split,
        dataset,
        bpe,
        src_dict,
        tgt_dict=None,
        max_src_length=128,
        max_tgt_length=30,
        patch_image_size=224,
        imagenet_default_mean_and_std=False,
        scst=False,
        diff_dataset=None,
        auxiliary_csv_path=None
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.patch_image_size = patch_image_size
        self.scst = scst
        self.diff_dataset = diff_dataset

        self.transtab = str.maketrans({key: None for key in string.punctuation})

        if imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((patch_image_size, patch_image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        if type(bpe).__name__ == 'GPT2BPE':
            self.prompt = " what does the image describe?"
        elif type(bpe).__name__ == 'BertBPE':
            self.prompt = "图片描述了什么内容?"

        if auxiliary_csv_path is not None:
            aux_df = pd.read_csv(auxiliary_csv_path)

            # Convert string representations of list columns to actual lists.
            columns_to_convert = ['disease_vector']
            for col in columns_to_convert:
                aux_df[col] = aux_df[col].apply(ast.literal_eval)

            # Get disease frequencies, sorted in descending order (most frequent first)
            disease_freq = aux_df['disease'].dropna().value_counts()

            # unique_diseases now is a list of diseases ordered by frequency
            unique_diseases = list(disease_freq.index)

            # Build the mapping from disease to index
            self.disease_to_index = {disease: idx for idx, disease in enumerate(unique_diseases)}

            # Group rows by dicom_id so that images with multiple disease entries are combined.
            grouped = aux_df.groupby("dicom_id")

            auxiliary_dict = {}
            for dicom_id, group in grouped:
                # For the disease vector, simply take it from the first row.
                disease_vector = np.array(group.iloc[0]['disease_vector'], dtype=np.float32)

                # Mask for getting the information from the reference image
                ref_id_mask = group.iloc[0]["ref_id_mask"]

                # Extract the reference DICOM ID from the first row.
                ref_dicom_id = group.iloc[0]["ref_dicom_id"]
                disease_present = 1

                # Update the matrices for each disease entry in this group.
                for _, row in group.iterrows():
                    # Skip rows where disease label is missing.
                    if pd.isna(row["disease"]):
                        disease_present = 0
                        continue
                    disease_label = row["disease"]
                    if disease_label not in self.disease_to_index:
                        print(f"CaptionDataset init method: Non existing label {disease_label} " 
                              f"for auxiliary dataset {auxiliary_csv_path}")
                        continue  # Defensive check

                auxiliary_dict[dicom_id] = {
                    "disease_vector_gt": disease_vector,  # shape: [disease_vector_dim]
                    "disease_present": disease_present,
                    "ref_dicom_id": ref_dicom_id,         # identifier string of reference image
                    "ref_id_mask": ref_id_mask      # 1 for cases ref_dicom_id info should be accessed 
                }                                   # by pointing to dicom_ids with -1 on ref_id_mask
                                                    # -1 for those additional entries  
                                                    # 0 for nothing special
                #print(f"{dicom_id}:\nLoc matrix: {loc_matrix}\nSev matrix:{sev_matrix}"
                #      f"\nDisease vector: {disease_vector}\nDisease present:{disease_present}"
                #      f"ref dicom: {ref_dicom_id}\nref mask: {ref_id_mask}")

            self.auxiliary_dict = auxiliary_dict
        else:
            self.auxiliary_dict = {}
            

    def process_data(self, index, dataset):
        uniq_id, image, caption, question, prev_image, _, dataset_name, _ = dataset[index%len(dataset)]

        image = Image.open(BytesIO(base64.urlsafe_b64decode(image)))
        patch_image = self.patch_resize_transform(image)
        patch_mask = torch.tensor([True])
        
        if self.split == 'train' and not self.scst:
            caption = caption.translate(self.transtab).strip()
            caption_token_list = caption.strip().split()
            tgt_caption = ' '.join(caption_token_list[:self.max_tgt_length])
        else:
            caption = ' '.join(caption.strip().split())
            caption_list = [cap.translate(self.transtab).strip() for cap in caption.strip().split('&&')]
            tgt_caption = '&&'.join(caption_list)
        
        tgt_item = self.encode_text(" {}".format(tgt_caption))
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])

        if dataset_name == 'mimic_f':
            src_item = self.encode_text(self.prompt)
            
        elif dataset_name == 'mimic_i':
            src_item = self.encode_text('what is the summary of the image?')     
              
        else:
            if self.split == 'train' and not self.scst:
                question = question.translate(self.transtab).strip()
                question_token_list = question.strip().split()
                tgt_question = ' '.join(question_token_list[:self.max_src_length])
            else:
                question = ' '.join(question.strip().split())
                question_list = [cap.translate(self.transtab).strip() for cap in question.strip().split('&&')]
                tgt_question = '&&'.join(question_list)
            src_item = self.encode_text(tgt_question)

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        
        
        example = {
            "id": uniq_id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item
        }

        # Add auxiliary data for the main image using uniq_id as key.
        aux_main = self.auxiliary_dict.get(uniq_id)
        if aux_main is not None:
            # Use the precomputed disease vector (a vector) and the ground truth matrices.
            example["disease_tensor_gt"] = torch.tensor(aux_main["disease_vector_gt"])
            example["disease_present"] =torch.tensor(aux_main["disease_present"])
        else:
            # If there's no auxiliary data for this image, default to all 0s (implicitly no disease).
            print(f"CaptionDataset: process_data: There is no auxiliary data for {uniq_id}")
            example["disease_tensor_gt"] = None
            example["disease_present"] = None

        # Process the reference image if available.
        if len(prev_image) != 0:
            prev_image = Image.open(BytesIO(base64.urlsafe_b64decode(prev_image)))
            prev_patch_image = self.patch_resize_transform(prev_image)
            example["patch_image_2"] = prev_patch_image

            if aux_main is not None:
                ref_dicom_id = aux_main["ref_dicom_id"]
                aux_ref = self.auxiliary_dict.get(ref_dicom_id)
                if aux_ref is not None:
                    example["disease_tensor_gt_2"] = torch.tensor(aux_ref["disease_vector_gt"])
                    example["ref_id_mask"] = torch.tensor(aux_ref["ref_id_mask"])
                    example["disease_present_2"] =torch.tensor(aux_ref["disease_present"])
                    #print(f"disease gt 2: {example['disease_tensor_gt_2']}")
                    #print(f"location gt 2: {example['location_tensor_gt_2']}")
                    #print(f"severity gt 2: {example['severity_tensor_gt_2']}")
                    #print(f"ref_id_mask: {example['ref_id_mask']}")
                    #print(f"disease_present_2: {example['disease_present_2']}")

        return [example]


    def __getitem__(self, index):
        example = self.process_data(index, self.dataset)

        diff_example = []
        if self.diff_dataset is not None:
            diff_example += self.process_data(index, self.diff_dataset)
        return example, diff_example


    def collater(self, samples, pad_to_length=None):
        """Merge samples of different tasks to form two mini-batches.
        Args:
            samples (List[Tuple]): samples to collate
        Returns:
            Tuple[dict]: two mini-batch containing the data of different tasks
        """

        samples_v1 = []   # containing image-text pairs
        samples_v2 = []   # containing detection data, text data and image data

        for sample_tuple in samples:
            samples_v1 += sample_tuple[0]
            samples_v2 += sample_tuple[1]
        if samples_v2 != []:
            res_v1 = collate(samples_v1, pad_idx=self.pad, eos_idx=self.eos)
            res_v2 = collate(samples_v2, pad_idx=self.pad, eos_idx=self.eos)
            return [res_v1, res_v2]
        else:
            res_v1 = collate(samples_v1, pad_idx=self.pad, eos_idx=self.eos)
            return res_v1
