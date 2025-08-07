import torch
import torch.nn as nn
from data import data_utils  # assumes collate_tokens is here

class PromptBuilder(nn.Module):
    def __init__(self, dictionary, task_bpe):
        super(PromptBuilder, self).__init__()
        self.dictionary = dictionary
        self.task_bpe = task_bpe
        # Pre-encode fixed tokens once (created on CPU initially)
        self.register_buffer('bos_tokens', torch.LongTensor([self.dictionary.bos()]))
        self.register_buffer('eos_tokens', torch.LongTensor([self.dictionary.eos()]))
        print(f"bos token: {self.bos_tokens}, eos token: {self.eos_tokens}")
        print(f"pad token: {self.dictionary.pad()}")

        # Pre-encode the vocabulary words:
        #self.disease_vocab = [
        #    'atelectasis', 'pleural effusion', 'lung opacity', 'edema', 
        #    'cardiomegaly', 'pneumonia', 'consolidation', 'vascular congestion', 
        #    'pneumothorax', 'infection', 'calcification', 'fracture', 'emphysema',
        #    'enlargement of the cardiac silhouette', 'pleural thickening', 
        #    'blunting of the costophrenic angle', 'hernia', 'scoliosis', 
        #    'heart failure', 'granuloma', 'pneumomediastinum', 'air collection', 
        #    'hilar congestion', 'hematoma', 'contusion', 'tortuosity of the thoracic aorta', 
        #    'gastric distention', 'hypoxemia', 'tortuosity of the descending aorta', 
        #    'hypertensive heart disease', 'thymoma'
        #]
        self.disease_vocab = [
            'atelectasis', 'pleural effusion', 'lung opacity', 'edema', 
            'cardiomegaly', 'pneumonia', 'consolidation', 'pneumothorax', 
            #'fracture', 'emphysema', 'pleural thickening', 'hernia', 
        ]
        # Pre-encode each vocabulary word:
        self.pre_enc_disease = [
            self.dictionary.encode_line(line=self.task_bpe.encode(word),
                                          add_if_not_exist=False,
                                          append_eos=False).long() 
            for word in self.disease_vocab
        ]
       
    def build_and_prepend_context(
            self,
            disease_main, disease_ref,
            src_tokens  # encoded question of shape: [B, max_seq_len]
        ):
        # Use the device of src_tokens for everything.
        device = src_tokens.device
        pad_idx = self.dictionary.pad()
        if disease_main.dim() > 1:
            B = disease_main.shape[0]
        else: 
            B = 1
        prompt_tokens_list = []
        
        context_mask_list = []
        #print(f'bos tokens: {self.bos_tokens}\neos tokens: {self.eos_tokens}')
        for i in range(B):
            # Compute the disease difference per sample.
            if B == 1:
                disease_diff = disease_main - disease_ref
            else:
                disease_diff = disease_main[i] - disease_ref[i]
            additional_words = []
            missing_words = []
            
            for j in range(disease_diff.shape[0]):
                val = disease_diff[j].item()
                if val == 1:
                    additional_words.append(self.disease_vocab[j])
                elif val == -1:
                    missing_words.append(self.disease_vocab[j])
            
            # Build string segments for additional and missing contexts.
            if additional_words:
                if len(additional_words) == 1:
                    additional_context_str = " additional " + additional_words[0]
                else:
                    additional_context_str = " additional " + " ".join(additional_words[:-1]) + " and " + additional_words[-1]
            else:
                additional_context_str = ""
            
            if missing_words:
                if len(missing_words) == 1:
                    missing_context_str = " missing " + missing_words[0]
                else:
                    missing_context_str = " missing " + " ".join(missing_words[:-1]) + " and " + missing_words[-1]
            else:
                missing_context_str = ""
            
            # If no changes, use the no-disease change string.
            # FIXME: This lines are commented for the reduced disease case
            #if not additional_words and not missing_words:
            #    context_str = " there is no change in diseases"
            #else:
            if not additional_words and not missing_words:
                context_mask_list.append(0)
                final_prompt = src_tokens[i]
            else:
                context_str = additional_context_str + missing_context_str + " "
                #context_str += " the question is "
                #print(f'context str number {i}: {context_str}')
                # Encode the prompt context (only the context; the question is already encoded)
                encoded_context = self.dictionary.encode_line(
                    line=self.task_bpe.encode(context_str),
                    add_if_not_exist=False,
                    append_eos=False
                ).long().to(device)
                # Prepend bos and append eos to the encoded prompt context
                prompt_encoded = torch.cat([self.bos_tokens.to(device), encoded_context])
                context_mask_list.append(prompt_encoded.shape[0])

                # Get the already encoded question tokens (which already include eos)
                question_encoded = src_tokens[i][src_tokens[i] != pad_idx]
                # Concatenate the prompt tokens with the encoded question.
                final_prompt = torch.cat([prompt_encoded, question_encoded[1:]])
            #print(f'encoded prompt number {i}:{final_prompt}')
            prompt_tokens_list.append(final_prompt)
        
        final_encoded = data_utils.collate_tokens(prompt_tokens_list, pad_idx)
        # build a matrix of 0s with the same shape as final_encoded
        context_mask = torch.zeros_like(final_encoded, dtype=torch.bool)
        # Modify the first context_mask_list[i] elements of each row to 2 with vectorized operation
        for i in range(B):
            context_mask[i, :context_mask_list[i]] = True
        # Debug: print the decoded prompt for inspection.
        # self.decode_full_prompt(final_encoded.detach())
        return final_encoded, context_mask
    
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
            bpe_symbol=None,          # don’t strip BPE markers
            extra_symbols_to_ignore=[],  # override defaults so it keeps special tokens
            unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP")
        )
        print(f"Raw decoded str in subwords:{decoded_str}")
        decoded_str = self.task_bpe.decode(decoded_str)
        print(f"Human readable decoded string in words:{decoded_str}")
