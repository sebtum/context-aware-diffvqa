import os
import json
import time
from datetime import datetime
import torch
import numpy as np
from PIL import Image, ImageOps
from io import BytesIO
import base64
from pycocotools.coco import COCO
from pycocoevalcap_custom.eval import COCOEvalCap
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from torchvision import transforms
from tasks.mm_tasks.refcoco import RefcocoTask
import csv

# Constants
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

# Main Class for Model Evaluation
class PLURALInference:
    def __init__(self, workspace, test_data, model_path, inference_path, annotation_file, results_file):
        self.workspace = workspace
        self.test_data = test_data
        self.model_path = model_path
        self.inference_path = inference_path
        self.annotation_file = annotation_file
        self.results_file = results_file
        self.use_cuda = torch.cuda.is_available()
        self.use_fp16 = False
        self.models = None
        self.task = None
        self.generator = None
        self.start_time = datetime.now()

    # Initialize the model and task
    def initialize_model(self):
        input_args = [
            "", "--task=refcoco", "--beam=10",
            f"--path={self.model_path}",
            f"--bpe-dir={self.workspace}/PLURAL/utils/BPE",
            "--no-repeat-ngram-size=3", "--patch-image-size=384"
        ]
        tasks.register_task('refcoco', RefcocoTask)
        parser = options.get_generation_parser()
        args = options.parse_args_and_arch(parser, input_args)
        cfg = convert_namespace_to_omegaconf(args)
        # Load pretrained ckpt and config
        self.task = tasks.setup_task(cfg.task)
        self.models, cfg = checkpoint_utils.load_model_ensemble(
            utils.split_paths(cfg.common_eval.path),
            task=self.task
        )
        # Move model to GPU
        for model in self.models:
            model.eval()
            if self.use_fp16:
                model.half()
            if self.use_cuda and not cfg.distributed_training.pipeline_model_parallel:
                model.cuda()
            model.prepare_for_inference_(cfg)

        self.generator = self.task.build_generator(self.models, cfg.generation)

    # Load and preprocess test data
    def load_data(self):
        with open(self.test_data, "r") as fp:
            fp.readline()  # Skip header
            data = []
            for line in fp:
                columns = line.rstrip("\n").split("\t")
                if len(columns) == 1:
                    break
                data.append({
                    "image": columns[1],
                    "gt": columns[2],
                    "question": columns[3],
                    "dataset": columns[6],
                    "prev_image": columns[4]
                })
        return data
    
    def load_data_generator(self):
        with open(self.test_data, "r") as fp:
            reader = csv.reader(fp, delimiter="\t")
            next(reader)  # Skip header
            for line in reader:
                yield {
                    "image": line[1],
                    "gt": line[2],
                    "question": line[3],
                    "dataset": line[6],
                    "prev_image": line[4]
                }


    # Construct a sample for inference
    def construct_sample(self, image, image2=None, instruction=""):
        patch_resize_transform = transforms.Compose([
            lambda img: img.convert("RGB"),
            transforms.Resize((self.task.cfg.patch_image_size, self.task.cfg.patch_image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])

        instruction = self.encode_text(instruction.lower().strip(), append_bos=True, append_eos=True).unsqueeze(0)
        instruction_length = torch.LongTensor([s.ne(self.task.src_dict.pad()).long().sum() for s in instruction])

        sample = {
            "id": np.array(['42']),
            "net_input": {
                "src_tokens": instruction,
                "src_lengths": instruction_length,
                "patch_images": patch_resize_transform(image).unsqueeze(0),
                "patch_masks": torch.tensor([True]),
            }
        }

        if image2:
            sample["net_input"]["patch_images_2"] = patch_resize_transform(image2).unsqueeze(0)

        return sample

    # Encode text for the model
    def encode_text(self, text, append_bos=False, append_eos=False):
        line = [
            self.task.bpe.encode(f' {word.strip()}')
            if not word.startswith('<code_') and not word.startswith('<bin_') else word
            for word in text.split()
        ]
        line = ' '.join(line)
        s = self.task.src_dict.encode_line(line, add_if_not_exist=False, append_eos=False).long()
        if append_bos:
            s = torch.cat([torch.LongTensor([self.task.src_dict.bos()]), s])
        if append_eos:
            s = torch.cat([s, torch.LongTensor([self.task.src_dict.eos()])])
        return s

    # Run inference
    def run_inference(self, data):
        infer_res = {"res": [], "gts": []}
        for i, item in enumerate(self.load_data_generator):
            instruction = item["question"]
            gt = item["gt"]

            if item["prev_image"]:
                image1 = Image.open(BytesIO(base64.urlsafe_b64decode(item["image"])))
                image2 = Image.open(BytesIO(base64.urlsafe_b64decode(item["prev_image"])))
                sample = self.construct_sample(image1, image2, instruction)
            else:
                image1 = Image.open(BytesIO(base64.urlsafe_b64decode(item["image"])))
                sample = self.construct_sample(image1, None, instruction)

            sample = utils.move_to_cuda(sample) if self.use_cuda else sample
            sample = utils.apply_to_sample(self.apply_half, sample) if self.use_fp16 else sample

            with torch.no_grad():
                hypos = self.task.inference_step(self.generator, self.models, sample)
                tokens, _, _ = self.decode_fn(hypos[0][0]["tokens"])

            infer_res["res"].append(tokens)
            infer_res["gts"].append(gt)

            print(f'{i + 1} / {len(data)}', end='\r') #FIXME: Progress bar

        return infer_res

    # Decode model output
    def decode_fn(self, x, bpe=None, tokenizer=None):
        decoded_str = self.task.src_dict.string(x.int().cpu(), extra_symbols_to_ignore=get_symbols_to_strip_from_output(self.generator))
        token_result, bin_result, img_result = [], [], []

        for token in decoded_str.strip().split():
            if token.startswith('<bin_'):
                bin_result.append(token)
            elif token.startswith('<code_'):
                img_result.append(token)
            else:
                if bpe is not None:
                    token = bpe.decode('{}'.format(token))
                if tokenizer is not None:
                    token = tokenizer.decode(token)
                if token.startswith(' ') or len(token_result) == 0:
                    token_result.append(token.strip())
                else:
                    token_result[-1] += token

        return ' '.join(token_result), ' '.join(bin_result), ' '.join(img_result)

    # Save results as JSON
    def save_results(self, results):
        res = [{"image_id": str(i), "caption": r} for i, r in enumerate(results["res"])]
        gts = {"images": [{"id": str(i), "caption": g} for i, g in enumerate(results["gts"])]}
        json.dump(res, open(f'{self.inference_path}/{self.results_file}', 'w'))
        json.dump(gts, open(f'{self.inference_path}/{self.annotation_file}', 'w'))

    # Evaluate results using COCO metrics
    def evaluate_results(self):
        coco = COCO(f'{self.inference_path}/{self.annotation_file}')
        coco_result = coco.loadRes(f'{self.inference_path}/{self.results_file}')
        coco_eval = COCOEvalCap(coco, coco_result)
        coco_eval.params['image_id'] = coco_result.getImgIds()
        coco_eval.evaluate()
        for metric, score in coco_eval.eval.items():
            print(f'{metric}: {score:.3f}')

    # Apply half-precision if needed
    @staticmethod
    def apply_half(t):
        return t.half() if t.dtype == torch.float32 else t

# Main execution
def main():
    timestamp = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    workspace = "/home/guests/sebastian_sotela"
    inference_path = f"{workspace}/logs/reportgen/inference"
    test_data_file = "mimic_diff_qa_s2_test.tsv"
    test_data = f"{workspace}/data_pre_runs/tsv_output_data/{test_data_file}"
    checkpoint_file = "PLURAL.pt"
    model_path = f"/home/polyaxon-data/data1/mimic/PLURAL/checkpoints/{checkpoint_file}"
    annotation_file = f"gts_{checkpoint_file[:-4]}_{test_data_file[6:-5]}_{timestamp}.json"
    results_file = f"res_{checkpoint_file[:-4]}_{test_data_file[6:-5]}_{timestamp}.json"

    inference = PLURALInference(workspace, test_data, model_path, inference_path, annotation_file, results_file)
    inference.initialize_model()
    data = inference.load_data()
    results = inference.run_inference(data)
    inference.save_results(results)
    inference.evaluate_results()
    print(f"Execution Time: {datetime.now() - inference.start_time}")

if __name__ == "__main__":
    main()
