import csv
import os
from PIL import Image
from io import BytesIO
import base64
import pandas as pd
import random
import numpy as np
from tqdm import tqdm
import logging
import concurrent.futures
from datetime import datetime
import asyncio
import time
import aiofiles
from aiofiles import open as aio_open 

def main():
    ws = '/home/guests/sebastian_sotela'
    output_dir = f'{ws}/data_pre_runs'
    data_dir = f'{output_dir}/tsv_output_data'

    # Configure logging
    timestamp = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    logging.basicConfig(
        filename=f"{output_dir}/logs/image_data_preprocessing_{timestamp}.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    root = '/home/polyaxon-data/data1/mimic'
    mimic_cxr_jpg = f'{root}/mimic-cxr-jpg/2.0.0'
    metadata_file = f'{mimic_cxr_jpg}/mimic-cxr-2.0.0-metadata.csv'
    images_path = f'{mimic_cxr_jpg}/files'
    diff_vqa_path = f'{root}/diff_vqa/physionet.org/files/medical-diff-vqa/1.0.0'
    qa_file = f'{diff_vqa_path}/mimic_pair_questions.csv'
    mimic_all_file = f'{diff_vqa_path}/mimic_all.csv'

    df_metadata = pd.read_csv(metadata_file)[['ViewPosition', 'dicom_id']]
    df_qa = pd.read_csv(qa_file)
    df_qa = df_qa[df_qa["question_type"] == "difference"]\
        [['subject_id', 'study_id', 'ref_id', 'question', 'answer', 'split']]
    df_mimic_all = pd.read_csv(mimic_all_file)[['study_id', 'dicom_id']]

    #tasks = ['findings', 'impressions', 'diff_qa_s2', 'diff_qa']
    tasks = [ 'findings' ] #FIXME findings training is incomplete
    #splits = ['valid', 'test', 'train']
    splits = ['train']

    for task in tasks:
        for split in splits:
            output_name = f'mimic_{task}_{split}_{timestamp}'
            tsv_file = f'{data_dir}/{output_name}.tsv'

            if 'qa' in task:
                logging.info(f'Processing task {task} and split {split}')
                chunk_size = 500
                write_buffer_size = 25
                dataset_name = 'mimic_qa'

                if task == 'diff_qa_s2':
                    # general difference question
                    df1 = df_qa[
                        df_qa["question"] == "what has changed compared to the reference image?"
                    ]
                else: # task == 'diff_qa'
                    # 23 specific questions of type difference
                    df1 = df_qa[
                        df_qa["question"] != "what has changed compared to the reference image?"
                    ]
                df2 = df_mimic_all
                # merge to add dicom_id
                merged_df = pd.merge(df1, df2, on='study_id', how='left')
                #merge to add ref id
                merged_df = pd.merge(merged_df, df2, left_on='ref_id', right_on='study_id', how='left', suffixes=('', '_ref'))
                # Drop duplicate study_id column
                merged_df.drop(columns=['ref_id'], inplace=True)

                if split =='valid':
                    filtered_df = merged_df[merged_df['split'] == 'val']
                else:
                    filtered_df = merged_df[merged_df['split'] == split]

            else: # task!='qa'
                # test split is merged with validation split
                if split == 'test':
                    continue
                logging.info(f'Processing task {task} and split {split}')
                chunk_size = 1000
                write_buffer_size = 50
                data_file = f'{ws}/mimic_data/mimic_{split}_{task}.csv'
                df = pd.read_csv(data_file)
                logging.info(f'reading {data_file}')            

                if split == 'valid':
                    # Combine test and validation for impressions and findings
                    df_test = pd.read_csv(f'{ws}/mimic_data/mimic_test_{task}.csv')
                    logging.info(f'Combined validation and test dataset for task: {task}. Previous size:'
                          f'{df.shape[0]}')
                    df = pd.concat([df, df_test], ignore_index = True)
                    logging.info(f'New size: {df.shape[0]}')

                if task == 'findings':
                    question = 'what does the image describe?'
                    dataset_name = 'mimic_f'
                    # Filter nonsense reports based on regex patterns
                    re_patterns = [r'((is|were|are) submitted (\.|dated))|(is submitted for interpretation \.)', r'chest (film|radiograph)  at ']
                    combined_pattern = f"({'|'.join(re_patterns)})"
                    no_regex_pattern_df = df[~df['report'].str.contains(combined_pattern, na=False, regex=True)]
                    logging.info(f'Filter nonsense reports (regex pattern) for task {task}. Previous size:'
                          f'{df.shape[0]}. New size: {no_regex_pattern_df.shape[0]}')
                    patterns = [' .', 'is .', 'am .', 'be .', 'in .', 'at the .', 'see below .', 'send the .', 'see below .',
                             ' through  .', 'is and for .', 'ct chest of  .', 'chest radiograph  .', 'no previous images .', 
                             'see impression below .', 'prior chest radiographs  .', 'portable chest radiograph .', 
                             'ct scan of the torso from  .','radiographs are under penetrated .','portable ap upright chest  at 1722 .',
                             'prior chest radiographs  through 47 .','portable ap approach vessel  at 0739 .',
                             'chest radiographs since  most recently  .','prior chest radiographs  through  at 1444 .',
                             'pa and lateral views of the chest  at 0923 .','portable semi-upright chest radiograph  0555 .',
                             'prior chest radiographs since  most recently  .', 'ap and lateral chest radiographs were obtained .',
                             'chest radiograph obtained approximately 3 hr prior .', ' ct torso .']

                else:
                    #task == impressions
                    question = 'what is the summary of the image?'
                    dataset_name = 'mimic_i'
                    patterns = [' .','as above .', 'see above .', 'as compared to  .']
                    no_regex_pattern_df = df
                
                # Filter nonsense reports based on fixed patterns
                no_fixed_pattern_df = no_regex_pattern_df[~no_regex_pattern_df['report'].isin(patterns)] 
                logging.info(f'Filter nonsense reports (fixed pattern) for task {task}. Previous size:'
                          f'{no_regex_pattern_df.shape[0]}. New size: {no_fixed_pattern_df.shape[0]}')

                # Filter test data from pair question
                test_ids = df_qa[df_qa['split']=='test']
                dicom_id_test = pd.merge(test_ids[['study_id']], df_mimic_all, on='study_id', how='inner')['dicom_id']
                clean_df = no_fixed_pattern_df[~no_fixed_pattern_df['dicom_id'].isin(dicom_id_test)]

                logging.info(f'Remove vqa test data for: {task}. Previous size:'
                          f'{no_fixed_pattern_df.shape[0]}. New size: {clean_df.shape[0]}')

                # Filter anything that is not antero posterior or postero anterior view
                dicom_ids = df_metadata[df_metadata['ViewPosition'].isin(['PA', 'AP'])]['dicom_id']
                filtered_df = clean_df[clean_df['dicom_id'].isin(dicom_ids)]
                logging.info(f'Keep only AP and PA views for task: {task}. Previous size:'
                          f'{clean_df.shape[0]}. New size: {filtered_df.shape[0]}')

            process_dataset_with_threads(filtered_df, images_path, task, dataset_name, tsv_file, chunk_size, write_buffer_size)
            logging.info(f"Finished processing {split} split for task {task}.")  

def process_dataset_with_threads(df, images_path, task, dataset_name, tsv_file, chunk_size=1000, max_workers=8, write_buffer_size=100):
    """
    Process a dataset in shuffled chunks using multithreading with async I/O for image processing.
    """
    # Shuffle the DataFrame rows
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df_size = len(df)
    logging.info(f'Size of the data frame {df_size}')
    #df_size = 2

    # Write TSV header if not already written
    if not os.path.exists(tsv_file):
        with open(tsv_file, 'w', newline='', encoding='utf-8') as tsv_out:
            tsv_writer = csv.writer(tsv_out, delimiter='\t')
            tsv_writer.writerow(['dicom_id', 'base64_str', 'report', 'question', 'ref_base64_str', '', 'dataset_name', 'task'])

    
    # Process the DataFrame in chunks
    for chunk_start in range(0, df_size, chunk_size):
        chunk_end = min(chunk_start + chunk_size, df_size)
        chunk_df = df.iloc[chunk_start:chunk_end]  # Slice the chunk
        if chunk_end % 2000 == 0:
            logging.info(f"Processing chunk {chunk_start}-{chunk_end} for task {task}")
        rows = list(chunk_df.itertuples(index=False))  # Convert chunk to namedtuples
        buffer = []  # Initialize buffer for results
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_row, row, images_path, task, dataset_name)
                for row in rows
            ]
            for future in futures:
                try:
                    result = future.result()
                    if result:
                        buffer.append(result)
                        # Write buffer when it reaches the specified size
                        if len(buffer) >= write_buffer_size:
                            asyncio.run(write_tsv_buffer(tsv_file, buffer))
                            buffer.clear()  # Clear the buffer after writing
                except Exception as e:
                    logging.error(f"Error processing row: {e}")

    # Write any remaining rows in the buffer
    if buffer:
        asyncio.run(write_tsv_buffer(tsv_file, buffer))
        buffer.clear()
    
async def write_tsv_buffer(tsv_file, buffer):
    """
    Asynchronously write the buffer to a TSV file.
    """
    async with aiofiles.open(tsv_file, mode='a', encoding='utf-8', newline='') as file:
        for row in buffer:
            line = '\t'.join(map(str, row)) + '\n'  # Convert row to TSV format
            await file.write(line)  # Asynchronous write

def process_row(row, images_path, task, dataset_name):
    """
    Process a single row, using async I/O for image processing.
    """
    async def process():
        dicom_id = row.dicom_id
        study_id = str(row.study_id)
        patient_id = str(row.subject_id)

        # Process the main image
        image_path = os.path.join(images_path, f'p{patient_id[:2]}/p{patient_id}/s{study_id}/{dicom_id}.jpg')
        base64_str = await async_retry(process_image_async, retries=3, delay=2, image_path=image_path)

        if 'qa' in task:
            ref_dicom_id = row.dicom_id_ref
            ref_study_id = str(int(row.study_id_ref))
            ref_image_path = os.path.join(images_path, f'p{patient_id[:2]}/p{patient_id}/s{ref_study_id}/{ref_dicom_id}.jpg')
            ref_base64_str = await async_retry(process_image_async, retries=3, delay=2, image_path=ref_image_path)
            question = row.question
            report = row.answer
        else:
            ref_base64_str = ''
            report = row.report
            question = 'What does the image describe?' if task == 'findings' else 'What is the summary of the image?'

        return [dicom_id, base64_str, report, question, ref_base64_str, '', dataset_name, 'caption']

    # Run async processing in the current thread
    return asyncio.run(process())

async def process_image_async(image_path):
    """
    Asynchronously processes a single image and converts it to Base64.
    """
    try:
        # Open the image file asynchronously
        async with aio_open(image_path, mode='rb') as img_file:
            byte_data = await asyncio.wait_for(img_file.read(), timeout=20)  # 10 seconds
            return base64.b64encode(byte_data).decode("utf-8")
    except asyncio.TimeoutError:
        logging.error(f"Timeout while reading: {image_path}")
        return None
    except FileNotFoundError:
        logging.error(f"File not found: {image_path}")
        return None
    except IOError as e:
        logging.error(f"I/O error for {image_path}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error for {image_path}: {e}")

async def async_retry(operation, retries=3, delay=2, *args, **kwargs):
    """
    Retry a given asynchronous operation with a specified number of attempts and delay.
    """
    for attempt in range(retries):
        try:
            return await operation(*args, **kwargs)
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed for operation {operation.__name__}: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(delay)  # Wait before retrying
            else:
                logging.error(f"Max retries reached for operation {operation.__name__}")
                raise

if __name__ == "__main__":
    main()