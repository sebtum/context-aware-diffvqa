import os
import json
import time
from datetime import datetime
import torch
import numpy as np
from PIL import Image, ImageOps
from io import BytesIO
import base64
from pycocoevalcap_custom.eval import COCOEvalCap
from pycocotools.coco import COCO
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from torchvision import transforms
from tasks.mm_tasks.refcoco import RefcocoTask
import csv
from concurrent.futures import ThreadPoolExecutor
import sys
import logging
import psutil
import threading

# Constants
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

def configure_logging():
    """Set up logging with stdout and stderr separation."""
    stdout_handler = logging.StreamHandler(sys.stdout)
    stderr_handler = logging.StreamHandler(sys.stderr)
    # Separate log levels for handlers
    stdout_handler.setLevel(logging.INFO)  # INFO and DEBUG logs go to stdout
    stderr_handler.setLevel(logging.ERROR)  # ERROR and CRITICAL logs go to stderr
    # Format for logs
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stdout_handler.setFormatter(formatter)
    stderr_handler.setFormatter(formatter)
    # Configure the logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Capture all log levels
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)
    # Log initialization details
    logger.info("Logging configured with stdout and stderr separation.")
    return logger

# Main Class for Model Evaluation
class PLURALInference:
    def __init__(self, workspace, test_data, model_path, inference_path, annotation_file, results_file, batch_size, num_workers, write_chunk_size):
        self.logger = logging.getLogger(__name__)
        self.workspace = workspace
        self.test_data = test_data
        self.model_path = model_path
        self.inference_path = inference_path
        self.annotation_file = annotation_file
        self.results_file = results_file
        self.batch_size = batch_size #FIXME: repeated unused
        self.num_workers = num_workers
        self.write_chunk_size = write_chunk_size
        self.use_cuda = torch.cuda.is_available()
        self.use_fp16 = False
        self.models = None
        self.task = None
        self.generator = None
        self.start_time = datetime.now()
        # Increase the field size limit
        csv.field_size_limit(21 * 1024 * 1024)  # 21 MB limit (approx 20 million characters)
        self.logger.info(f"PLURALInference initialized with batch size={self.batch_size} and num_workers={self.num_workers}")
        self.logger.info(f"CSV field size limit set to: {csv.field_size_limit()}")
    
    def log_memory_usage(self, tag=""):
        """Log memory usage of the current process."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        self.logger.info(f"[{tag}] Memory Usage: {memory_info.rss / (1024 ** 2):.2f} MB")  # Resident memory in MB
    
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
        self.logger.info("Initializing model and task...")
        try:
            # Model initialization code...
            self.task = tasks.setup_task(cfg.task)
            self.models, cfg = checkpoint_utils.load_model_ensemble(
                utils.split_paths(cfg.common_eval.path),
                task=self.task
            )
            self.logger.info("Model and task successfully initialized.")
        except Exception as e:
            self.logger.error(f"Error during model initialization: {e}", exc_info=True)
            raise
        # Move model to GPU
        for model in self.models:
            model.eval()
            if self.use_fp16:
                model.half()
            if self.use_cuda and not cfg.distributed_training.pipeline_model_parallel:
                model.cuda()
            model.prepare_for_inference_(cfg)
        try:
            self.logger.info("Building generator...")
            self.generator = self.task.build_generator(self.models, cfg.generation)
            self.logger.info("Successfully built generator")
        except Exception as e:
            self.logger.error(f"Error during generator buildup: {e}", exc_info=True)
            raise

    def load_data_generator(self):
        self.logger.info(f"Starting to load data from: {self.test_data}")
        line_count = 0
        try:
            with open(self.test_data, "r") as fp:
                reader = csv.reader(fp, delimiter="\t")
                next(reader)  # Skip header
                for line in reader:
                    line_count += 1
                    yield {
                        "image": line[1],
                        "gt": line[2],
                        "question": line[3],
                        "dataset": line[6],
                        "prev_image": line[4]
                    }
        except Exception as e:
            self.logger.error(f"Error loading data at line {line_count}: {e}")
        finally:
            self.logger.info(f"Finished loading {line_count} lines of data.")
    
    # Construct a batch for inference
    def construct_batch(self, batch_data):
        patch_resize_transform = transforms.Compose([
            lambda img: img.convert("RGB"),
            transforms.Resize((self.task.cfg.patch_image_size, self.task.cfg.patch_image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])

        samples, gts = [], []
        for i, item in enumerate(batch_data):
            try:
                instruction = self.encode_text(item["question"].lower().strip(), append_bos=True, append_eos=True).unsqueeze(0)
                instruction_length = torch.LongTensor([s.ne(self.task.src_dict.pad()).long().sum() for s in instruction])

                image1 = Image.open(BytesIO(base64.urlsafe_b64decode(item["image"])))
                patch_image = patch_resize_transform(image1).unsqueeze(0)
                image1.close()

                sample = {
                    "id": torch.LongTensor([42]),
                    "net_input": {
                        "src_tokens": instruction,
                        "src_lengths": instruction_length,
                        "patch_images": patch_image,
                        "patch_masks": torch.tensor([True]),
                    }
                }

                if item.get("prev_image"):
                    image2 = Image.open(BytesIO(base64.urlsafe_b64decode(item["prev_image"])))
                    sample["net_input"]["patch_images_2"] = patch_resize_transform(image2).unsqueeze(0)
                    image2.close()
                #self.logger.debug(f"Sample: {sample}")
                samples.append(sample)
                gts.append(item["gt"])

                # Clean up variables
                del instruction, instruction_length, patch_image, sample

            except KeyError as e:
                self.logger.error(f"Missing key in item {i + 1}: {e}", exc_info=True)
            except base64.binascii.Error as e:
                self.logger.error(f"Base64 decoding error in item {i + 1}: {e}", exc_info=True)
            except Exception as e:
                self.logger.error(f"Unexpected error in item {i + 1}: {e}", exc_info=True)

        return samples, gts

    # Encode text for the model
    def encode_text(self, text, append_bos=False, append_eos=False):
        line = [
            self.task.bpe.encode(f' {word.strip()}')
            if not word.startswith('<code_') and not word.startswith('<bin_') else word
            for word in text.strip().split()
        ]
        line = ' '.join(line)
        s = self.task.tgt_dict.encode_line(
            line=line,
            add_if_not_exist=False,
            append_eos=False
        ).long()
        if append_bos:
            s = torch.cat([torch.LongTensor([self.task.src_dict.bos()]), s])
        if append_eos:
            s = torch.cat([s, torch.LongTensor([self.task.src_dict.eos()])])
        del line
        return s
    
        # Process a single batch
    def process_batch(self, batch):
        #self.logger.debug("Constructing batch...")
        thread_id = threading.get_ident()
        #self.log_memory_usage(f"Processing batch and startin its construction with thread id {thread_id}")
        samples, gts = self.construct_batch(batch)
        #self.log_memory_usage(f"After constructing batch with thread id {thread_id}")
        #self.logger.debug(f"Constructed batch samples: {samples[:1]}")  # Log the first sample for inspection
        #self.logger.debug(f"Ground truths: {gts[:1]}")  # Log the first ground truth

        samples = utils.move_to_cuda(samples) if self.use_cuda else samples
        samples = utils.apply_to_sample(self.apply_half, samples) if self.use_fp16 else samples
        
        infer_results = {"res": [], "gts": []}

        for i, sample in enumerate(samples):
            try:
                with torch.no_grad():
                    #self.logger.debug("Running inference step...")
                    #self.log_memory_usage(f"Before inference tid {thread_id}")
                    hypos = self.task.inference_step(self.generator, self.models, sample)
                    #self.log_memory_usage(f"After inference tid {thread_id}")
                    #self.logger.debug(f"Inference output: {hypos[:1]}")  # Log first output for inspection
            except Exception as e:
                self.logger.error(f"Error during inference step: {e}", exc_info=True)
                raise
        
            try:
                with torch.no_grad():
                    #self.log_memory_usage(f"Before decoding tid {thread_id}")
                    tokens, _, _ = self.decode_fn(hypos[0][0]["tokens"], bpe=self.task.bpe)
                    #self.log_memory_usage(f"After decoding tid {thread_id}")
            except Exception as e:
                self.logger.error(f"Error during decoding of sample {i} in a batch.")
            infer_results["res"].append(tokens)
            infer_results["gts"].append(gts[i])
            #self.log_memory_usage(f"After appending inference results tid {thread_id}")
            # Clear tensors and free GPU memory
        del samples, gts, hypos, tokens
        torch.cuda.empty_cache()  # Clear GPU memory cache
        #self.log_memory_usage(f"After processing batch with thread id {thread_id}")
        return infer_results

    def run_parallel_inference(self):
        """Run inference in batches and write results in chunks."""
        start_time_inf = datetime.now()
        self.logger.info(f"Starting parallel inference with {self.num_workers} workers and batch size {self.batch_size}")
        batch_data = []  # Stores data for processing
        buffer = {"res": [], "gts": []} # Stores processed results for writing
        processed_batches = 0
        # chunk_counter = 0
        try:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []

                # Step 1: Process data in batches
                for item in self.load_data_generator():
                    batch_data.append(item)

                    # When processing batch size is reached, submit for processing
                    if len(batch_data) == self.batch_size:
                        if processed_batches % 10 == 0:
                            self.logger.debug(f"Submitting batch {processed_batches + 1}")
                        #self.log_memory_usage("Before appending and submiting batch...")
                        #futures.append(executor.submit(self.process_batch, batch_data))
                        futures.append(self.process_batch(batch_data)) #FIXME
                        #self.log_memory_usage("After appending and submitting batch...")
                        batch_data = []
                        #self.log_memory_usage("After clearing batch data")
                        processed_batches += 1

                # Process remaining data
                if batch_data:
                    #self.log_memory_usage("Before remaining batch processing...")
                    #futures.append(executor.submit(self.process_batch, batch_data))
                    futures.append(self.process_batch(batch_data))
                    processed_batches += 1

                # Step 2: Collect results and write in chunks
                for future in futures:
                    try:
                        #self.log_memory_usage("Before collecting batch results...")
                        #batch_results = future.result()  # Get results from process_batch
                        batch_results = future
                        #self.log_memory_usage("After collecting batch results...")
                        buffer["res"].extend(batch_results["res"])  # Extend tokens
                        buffer["gts"].extend(batch_results["gts"])  # Extend ground truths
                        #self.log_memory_usage("After appending to buffer...")
                        #self.logger.debug(f"Batch completed.")
                    except Exception as e:
                        self.logger.error(f"Error during batch processing: {e}", exc_info=True)

            if buffer:
                self.write_results(buffer)
                self.logger.info("Parallel inference completed successfully.")
                buffer = {"res": [], "gts": []}  # Clear the buffer
            else:
                self.logger.error(f"No data to be written. Empty buffer.")
                sys.exit(0)

        except Exception as e:
            self.logger.error(f"Critical error during parallel inference: {e}", exc_info=True)
            raise

        self.logger.info(f"Parallel Inference execution Time: {datetime.now() - start_time_inf}")

    def get_symbols_to_strip_from_output(self):
        if hasattr(self.generator, "symbols_to_strip_from_output"):
            sym2strip = self.generator.symbols_to_strip_from_output
            return sym2strip
        else:
            sym2strip = {self.generator.bos, self.generator.eos}
            return sym2strip
        
    # Decode model output
    def decode_fn(self, x, bpe=None, tokenizer=None):
        decoded_str = self.task.tgt_dict.string(x.int().cpu(), extra_symbols_to_ignore=self.get_symbols_to_strip_from_output())
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

    def write_results(self, infer_res):
        """Write the entire results and annotations in COCO format."""
        self.logger.info("Writing the entire results and annotations in COCO format.")
        res_file_path = f"{self.inference_path}/{self.results_file}"
        gts_file_path = f"{self.inference_path}/{self.annotation_file}"

        res = []
        gts_img = []
        gts_ann = []

        for i, cap in enumerate(zip(infer_res['res'], infer_res['gts'])):
            res.append({"image_id": str(i * 2), "caption": cap[0]})
            gts_img.append({"id": str(i * 2), "caption": cap[1]})
            gts_ann.append({"image_id": str(i * 2), "id": str(i), "caption": cap[1]})

        gts = {"images": gts_img, "annotations": gts_ann}

        # Write results and annotations
        try:
            with open(res_file_path, 'w') as res_fp:
                json.dump(res, res_fp)
            with open(gts_file_path, 'w') as gts_fp:
                json.dump(gts, gts_fp)
        except FileNotFoundError as e:
            self.logger.error(f"File not found error while writing: {e}", exc_info=True)
        except IOError as e:
            self.logger.error(f"I/O error while writing: {e}", exc_info=True)
        except Exception as e:
            self.logger.error(f"Unexpected error during file writing: {e}", exc_info=True)
            raise

        self.logger.info(f"Final results written to {res_file_path} and {gts_file_path}.")


    # Evaluate results using COCO metrics
    def evaluate_results(self, nltk_download_dir):
        start_time_eval = datetime.now()
        coco = COCO(f'{self.inference_path}/{self.annotation_file}')
        coco_result = coco.loadRes(f'{self.inference_path}/{self.results_file}')
        coco_eval = COCOEvalCap(coco, coco_result, nltk_download_dir)
        coco_eval.params['image_id'] = coco_result.getImgIds()
        self.logger.info("Starting evaluation...")
        try:
            coco_eval.evaluate()
            self.logger.info("Evaluation completed.")
        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}", exc_info=True)
            raise
        self.logger.info(f"NLP metrics evaluation Time: {datetime.now() - start_time_eval}")
        

    # Apply half-precision if needed
    @staticmethod
    def apply_half(t):
        return t.half() if t.dtype == torch.float32 else t

# Main execution
def main():
    logger = configure_logging()
    timestamp = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    in_seraph = os.getenv("IN_SERAPH", "0") == "1"
    if in_seraph:
        logger.info("Running on seraph cluster...")
        # NAS disk storage
        nas_path = "/data/geraugi"
        repo_ws = f"{nas_path}/repos"
        plural_ws = f"{nas_path}/plural"
        inference_path = f"{plural_ws}/inference"
        test_data_file = "mimic_diff_qa_test.tsv"
        #test_data = f"{plural_ws}/pre_processed_data/{test_data_file}"
        test_data = f"{plural_ws}/pre_processed_data/100_samples/{test_data_file}" #FIXME
        
        checkpoint_file = "PLURAL.pt"
        model_path = f"{plural_ws}/checkpoints/{checkpoint_file}"
        
        # checkpoint after full sequential training
        #checkpoint_file = "checkpoint_best.pt"
        #model_path = f"{plural_ws}/checkpoints/finetune/3_0.06_200000/{checkpoint_file}"
        
        # checkpoint before specific dvqa
        #checkpoint_file = "checkpoint_best.pt"
        #model_path = f"{plural_ws}/checkpoints/pretrain/3_0.06_200000_2025-01-09_09-21-55/{checkpoint_file}"
        
    
    else:
        logger.info("Running on ifl cluster...")
        workspace = "/home/guests/sebastian_sotela"
        inference_path = f"{workspace}/logs/reportgen/inference"
        test_data_file = "mimic_diff_qa_s2_test.tsv"
        test_data = f"{workspace}/data_pre_runs/tsv_output_data/100_samples/{test_data_file}" #FIXME
        checkpoint_file = "PLURAL.pt"
        model_path = f"/home/polyaxon-data/data1/mimic/PLURAL/checkpoints/{checkpoint_file}"
    
    just_score = False

    if just_score == True:
        # sequential general dvqa
        #annotation_file = "gts_checkpoint_best_diff_qa_s2_test_2025_01_10_11:53:06.json"
        #results_file = "res_checkpoint_best_diff_qa_s2_test_2025_01_10_11:53:06.json"
        # sequential specific dvqa
        #annotation_file = "gts_checkpoint_best_diff_qa_test_2025_01_10_08:30:10.json"
        #results_file = "res_checkpoint_best_diff_qa_test_2025_01_10_08:30:10.json"
        # plural benchmark general dvqa
        #annotation_file = "gts_PLURAL_diff_qa_s2_test_2024_12_14_22:50:13.json"
        #results_file = "res_PLURAL_diff_qa_s2_test_2024_12_14_22:50:13.json"
        # plural benchmark specific dvqa
        annotation_file = "gts_PLURAL_diff_qa_test_2024_12_15_02:27:49.json"
        results_file = "res_PLURAL_diff_qa_test_2024_12_15_02:27:49.json"
    else:
        annotation_file = f"gts_{checkpoint_file[:-3]}_{test_data_file[6:-4]}_{timestamp}.json"
        results_file = f"res_{checkpoint_file[:-3]}_{test_data_file[6:-4]}_{timestamp}.json"

    inference = PLURALInference(
        repo_ws, test_data, model_path, inference_path,
        annotation_file, results_file, batch_size=8, num_workers=1,
        write_chunk_size=500
    )
    if just_score == True:
        inference.evaluate_results(nas_path)
    else:
        inference.initialize_model()
        inference.run_parallel_inference()
        inference.evaluate_results(plural_ws)
        inference.logger.info(f"Execution Time: {datetime.now() - inference.start_time}")

if __name__ == "__main__":
    main()
