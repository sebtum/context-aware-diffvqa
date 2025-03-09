import torch
import torch.nn as nn
from data import data_utils  # assumes collate_tokens is here

class PromptBuilder(nn.Module):
    def __init__(self, dictionary, task_bpe):
        super(PromptBuilder, self).__init__()
        self.dictionary = dictionary
        self.task_bpe = task_bpe
        # Pre-encode fixed tokens once (created on CPU initially)
        self.bos_tokens = torch.LongTensor([self.dictionary.bos()])
        self.eos_tokens = torch.LongTensor([self.dictionary.eos()])
        self.main_prefix = self.dictionary.encode_line(
            line=self.task_bpe.encode("main image has "),
            add_if_not_exist=False,
            append_eos=False
        ).long()
        self.ref_prefix = self.dictionary.encode_line(
            line=self.task_bpe.encode(" reference image has "),
            add_if_not_exist=False,
            append_eos=False
        ).long()
        self.question_prefix = self.dictionary.encode_line(
            line=self.task_bpe.encode(" the question is "),
            add_if_not_exist=False,
            append_eos=False
        ).long()
        self.in_location = self.dictionary.encode_line(
            line=self.task_bpe.encode("in location "),
            add_if_not_exist=False,
            append_eos=False
        ).long()
        self.with_severity = self.dictionary.encode_line(
            line=self.task_bpe.encode("with severity "),
            add_if_not_exist=False,
            append_eos=False
        ).long()
        self.and_token = self.dictionary.encode_line(
            line=self.task_bpe.encode("and "),
            add_if_not_exist=False,
            append_eos=False
        ).long()
        self.not_specified = self.dictionary.encode_line(
            line=self.task_bpe.encode("not specified"),
            add_if_not_exist=False,
            append_eos=False
        ).long()
        self.no_disease = self.dictionary.encode_line(
            line=self.task_bpe.encode("no disease"),
            add_if_not_exist=False,
            append_eos=False
        ).long()
        self.space = self.dictionary.encode_line(
            line=self.task_bpe.encode(" "),
            add_if_not_exist=False,
            append_eos=False
        ).long()
        

        # Pre-encode the vocabulary words:
        self.disease_vocab = [
            'atelectasis', 'pleural effusion', 'lung opacity', 'edema', 
            'cardiomegaly', 'pneumonia', 'consolidation', 'vascular congestion', 
            'pneumothorax', 'infection', 'calcification', 'fracture', 'emphysema',
            'enlargement of the cardiac silhouette', 'pleural thickening', 
            'blunting of the costophrenic angle', 'hernia', 'scoliosis', 
            'heart failure', 'granuloma', 'pneumomediastinum', 'air collection', 
            'hilar congestion', 'hematoma', 'contusion', 'tortuosity of the thoracic aorta', 
            'gastric distention', 'hypoxemia', 'tortuosity of the descending aorta', 
            'hypertensive heart disease', 'thymoma'
        ]
        self.location_vocab = [
            'left', 'lower', 'middle', 'right', 'upper', 'lung', 'pericardial', 'pleural', 'retrocardiac', 'rib'
        ]
        self.severity_vocab = {
            1: 'mild',
            2: 'mild to moderate',
            3: 'moderate',
            4: 'moderate to severe',
            5: 'severe'
        }
        # Pre-encode each vocabulary word:
        self.pre_enc_disease = [
            self.dictionary.encode_line(line=self.task_bpe.encode(word),
                                          add_if_not_exist=False,
                                          append_eos=False).long() 
            for word in self.disease_vocab
        ]
        self.pre_enc_location = [
            self.dictionary.encode_line(line=self.task_bpe.encode(word),
                                          add_if_not_exist=False,
                                          append_eos=False).long() 
            for word in self.location_vocab
        ]
        self.pre_enc_severity = [
            self.dictionary.encode_line(line=self.task_bpe.encode(self.severity_vocab[i]),
                                          add_if_not_exist=False,
                                          append_eos=False).long() 
            for i in range(1, 6)
        ]
        
        self.threshold = 0.5

    def build_and_prepend_context(
            self,
            disease_logits, disease_logits_2,
            location_logits, location_logits_2,
            severity_logits, severity_logits_2,
            disease_tensor_gt, disease_tensor_gt_2,
            location_tensor_gt, location_tensor_gt_2,
            severity_tensor_gt, severity_tensor_gt_2,
            src_tokens  # encoded question of shape: [B, max_seq_len]
        ):
        # Use the device of src_tokens for everything.
        device = src_tokens.device

        # Move fixed tokens to the target device.
        bos_tokens = self.bos_tokens.to(device)
        eos_tokens = self.eos_tokens.to(device)
        main_prefix = self.main_prefix.to(device)
        ref_prefix = self.ref_prefix.to(device)
        question_prefix = self.question_prefix.to(device)
        in_location = self.in_location.to(device)
        with_severity = self.with_severity.to(device)
        and_token = self.and_token.to(device)
        not_specified = self.not_specified.to(device)
        no_disease = self.no_disease.to(device)
        space = self.space.to(device)

        # For teacher forcing, keep the targets as they are; otherwise compute predictions.
        if self.training:
            disease_main = disease_tensor_gt  # assumed to be on GPU
            location_main = location_tensor_gt
            severity_main = severity_tensor_gt
            disease_ref = disease_tensor_gt_2
            location_ref = location_tensor_gt_2
            severity_ref = severity_tensor_gt_2
        else:
            disease_main = (torch.sigmoid(disease_logits) > self.threshold).to(device)
            location_main = (torch.sigmoid(location_logits) > self.threshold).to(device)
            severity_main = (torch.sigmoid(severity_logits) > self.threshold).to(device)
            disease_ref = (torch.sigmoid(disease_logits_2) > self.threshold).to(device)
            location_ref = (torch.sigmoid(location_logits_2) > self.threshold).to(device)
            severity_ref = (torch.sigmoid(severity_logits_2) > self.threshold).to(device)

        B = disease_main.shape[0]
        prompt_tokens_list = []
        pad_idx = self.dictionary.pad()

        for i in range(B):
            main_tokens = []  # List of 1D tensors for main image context.
            if disease_main[i].sum().item() == 0:
                main_tokens.append(no_disease)
            else:
                for j in range(disease_main.shape[1]):    
                    if disease_main[i, j].item() == 1:
                        main_tokens.append(self.pre_enc_disease[j].to(device))
                        main_tokens.append(space)
                        # Location: check if any location is active for disease j.
                        if location_main[i, j].sum().item() > 0:
                            main_tokens.append(in_location)
                            for k in range(location_main.shape[2]):
                                if location_main[i, j, k].item() == 1:
                                    main_tokens.append(self.pre_enc_location[k].to(device))
                                    main_tokens.append(space)
                        else:
                            main_tokens.append(in_location)
                            main_tokens.append(not_specified)
                            main_tokens.append(space)
                        # Severity: if any severity is active.
                        if severity_main[i, j].sum().item() > 0:
                            main_tokens.append(with_severity)
                            indices = (severity_main[i, j] == 1).nonzero(as_tuple=False)
                            if indices.numel() > 0:
                                idx = int(indices.max())
                                main_tokens.append(self.pre_enc_severity[idx].to(device))
                        else:
                            main_tokens.append(with_severity)
                            main_tokens.append(not_specified)
                        main_tokens.append(space)
                        main_tokens.append(and_token)
                main_tokens = main_tokens[:-2]  # Remove trailing space and "and"
            main_context = torch.cat([bos_tokens, main_prefix] + main_tokens + [eos_tokens])

            ref_tokens = []  # Token list for reference image context.
            if disease_ref[i].sum().item() == 0:
                ref_tokens.append(no_disease)
            else:
                for j in range(disease_ref.shape[1]):
                    if disease_ref[i, j].item() == 1:
                        ref_tokens.append(self.pre_enc_disease[j].to(device))
                        ref_tokens.append(space)
                        if location_ref[i, j].sum().item() > 0:
                            ref_tokens.append(in_location)
                            for k in range(location_ref.shape[2]):
                                if location_ref[i, j, k].item() == 1:
                                    ref_tokens.append(self.pre_enc_location[k].to(device))
                                    ref_tokens.append(space)
                        else:
                            ref_tokens.append(in_location)
                            ref_tokens.append(not_specified)
                            ref_tokens.append(space)
                        if severity_ref[i, j].sum().item() > 0:
                            ref_tokens.append(with_severity)
                            indices = (severity_ref[i, j] == 1).nonzero(as_tuple=False)
                            if indices.numel() > 0:
                                idx = int(indices.max())
                                ref_tokens.append(self.pre_enc_severity[idx].to(device))
                        else:
                            ref_tokens.append(with_severity)
                            ref_tokens.append(not_specified)
                        ref_tokens.append(space)
                        ref_tokens.append(and_token)
                ref_tokens = ref_tokens[:-2]
            ref_context = torch.cat([bos_tokens, ref_prefix] + ref_tokens + [eos_tokens])

            # For the question, we assume src_tokens is already on GPU; remove padding tokens.
            question_tokens = src_tokens[i][src_tokens[i] != pad_idx]
            # Build the final prompt:
            # <bos> Main context <eos> <bos> Reference context <eos> Question (excluding first token) <eos>
            full_prompt = torch.cat([
                main_context, ref_context, bos_tokens, question_prefix,
                question_tokens[1:], eos_tokens
            ])
            prompt_tokens_list.append(full_prompt)
        final_encoded = data_utils.collate_tokens(prompt_tokens_list, pad_idx)
        # Ensure the final prompt tensor is on the same device as src_tokens.
        debug = False
        if debug:
            self.decode_full_prompt(final_encoded.detach())
        final_encoded = final_encoded.to(src_tokens.device)
        return final_encoded
    
    def decode_full_prompt(self, final_encoded, escape_unk=False):
        """
        Decodes the full prompt tensor into a human-readable string.

        Args:
            final_encoded (Tensor): The tensor output from build_and_prepend_context,
                typically of shape [B, T] where T is the sequence length.
            escape_unk (bool): If True, use a different unknown token string.

        Returns:
            A string representing the decoded prompt.
        """
        # Convert tensor to CPU and ensure it's integer type.
        token_ids = final_encoded.int().cpu()

        # Convert token IDs to a string using your dictionary.
        decoded_str = self.dictionary.string(
            token_ids,
            unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP")
        )
        print(f"Raw decoded str in subwords:{decoded_str}")
        decoded_str = self.task_bpe.decode(decoded_str)
        print(f"Human readable decoded string in words:{decoded_str}")
