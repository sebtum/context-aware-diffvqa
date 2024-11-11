import csv
import json
import os
from PIL import Image
from io import BytesIO
import base64
import pandas as pd
import pickle
import random
import numpy as np
from tqdm import tqdm
import logging

def main():
    # Configure logging
    logging.basicConfig(
        filename="image_data_preprocessing.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    ws = '/home/guests/sebastian_sotela'
    data_dir = f'{ws}/PLURAL/data'
    root = '/home/polyaxon-data/data1/mimic'
    mimic_cxr_jpg = f'{root}/mimic-cxr-jpg/2.0.0'
    metadata_file = f'{mimic_cxr_jpg}/mimic-cxr-2.0.0-metadata.csv'
    images_path = f'{mimic_cxr_jpg}/files'
    diff_vqa_path = f'{root}/diff_vqa/physionet.org/files/medical-diff-vqa/1.0.0'
    qa_file = f'{diff_vqa_path}/mimic_pair_questions.csv'
    mimic_all_file = f'{diff_vqa_path}/mimic_all.csv'

    df_metadata = pd.read_csv(metadata_file)
    df_qa = pd.read_csv(qa_file)
    df_mimic_all = pd.read_csv(mimic_all_file)

    tasks = ['findings', 'impressions', 'qa']
    splits = [ 'valid', 'test']
    # TODO 
    # loop, pickle file 

    for task in tasks:
        for split in splits:
            if task == 'qa':
                dataset_name = 'mimic_qa'

                df1 = df_qa[['subject_id', 'study_id', 'ref_id', 'question', 'answer', 'split']]
                df2 = df_mimic_all[['study_id', 'dicom_id']]
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

                df = pd.read_csv(f'{ws}/mimic_data/mimic_{split}_{task}.csv')            

                if split == 'valid':
                    # Combine test and validation for impressions and findings
                    df_test = pd.read_csv(f'{ws}/mimic_data/mimic_test_{task}.csv')
                    logging.info(f'Combined validation and test dataset for task: {task}.\nPrevious size:'
                          f'{df.shape[0]}\n')
                    df = pd.concat([df, df_test], ignore_index = True)
                    logging.info(f'New size: {df.shape[0]}\n')

                if task == 'findings':
                    question = 'what does the image describe?'
                    dataset_name = 'mimic_f'
                    # Filter nonsense reports based on regex patterns
                    re_patterns = [r'((is|were|are) submitted (\.|dated))|(is submitted for interpretation \.)', r'chest (film|radiograph)  at ']
                    combined_pattern = f"({'|'.join(re_patterns)})"
                    no_regex_pattern_df = df[~df['report'].str.contains(combined_pattern, na=False, regex=True)]
                    logging.info(f'Filter nonsense reports (regex pattern) for task {task}.\nPrevious size:'
                          f'{df.shape[0]}\nNew size: {no_regex_pattern_df.shape[0]}\n')
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
                
                # Filter nonsense reports based on fixed patterns
                no_fixed_pattern_df = no_regex_pattern_df[~no_regex_pattern_df['report'].isin(patterns)] 
                logging.info(f'Filter nonsense reports (fixed pattern) for task {task}.\nPrevious size:'
                          f'{no_regex_pattern_df.shape[0]}\nNew size: {no_fixed_pattern_df.shape[0]}\n')

                # Filter test data from pair question
                test_ids = df_qa[df_qa['split']=='test']
                dicom_id_test = pd.merge(test_ids[['study_id']], df_mimic_all[['study_id', 'dicom_id']], on='study_id', how='inner')['dicom_id']
                clean_df = no_fixed_pattern_df[~no_fixed_pattern_df['dicom_id'].isin(dicom_id_test)]

                logging.info(f'Remove vqa test data for: {task}.\nPrevious size:'
                          f'{no_fixed_pattern_df.shape[0]}\nNew size: {clean_df.shape[0]}\n')

                # Filter anything that is not antero posterior or postero anterior view
                dicom_ids = df_metadata[df_metadata['ViewPosition'].isin(['PA', 'AP'])]['dicom_id']
                filtered_df = clean_df[clean_df['dicom_id'].isin(dicom_ids)]
                logging.info(f'Keep only AP and PA views for task: {task}.\nPrevious size:'
                          f'{clean_df.shape[0]}\nNew size: {filtered_df.shape[0]}\n')

            contents = []
            logging.info(f'Processing task {task} and split {split}')

            for num_id in tqdm(range(len(filtered_df)), desc=f'Processing task {task} and split {split}', unit='images', leave=True):
                row = filtered_df.iloc[num_id]
                if num_id % 2000 == 0:
                    logging.info(f'{num_id} images have been processed.')
                #current image    
                dicom_id = row['dicom_id']
                study_id = str(row['study_id'])
                patient_id = str(row['subject_id'])    

                image_path = f'p{patient_id[:2]}/p{patient_id}/s{study_id}/{dicom_id}.jpg'
                image_path = os.path.join(images_path, image_path)
                base64_str = process_image(image_path)

                #ref image
                if task == 'qa':
                    ref_dicom_id = row['dicom_id_ref']
                    ref_study_id = str(int(row['study_id_ref']))
                    ref_patient_id = patient_id  

                    ref_image_path = f'p{ref_patient_id[:2]}/p{ref_patient_id}/s{ref_study_id}/{ref_dicom_id}.jpg'
                    ref_image_path = os.path.join(images_path, ref_image_path)
                    ref_base64_str = process_image(ref_image_path)

                    question = row['question']
                    report = row['answer']
                else:
                    ref_base64_str = ''
                    report = row['report'] 

                contents.append([str(num_id), base64_str, report, question, ref_base64_str,'', dataset_name,'caption'])

            random.shuffle(contents)
            f = open(f'{data_dir}/mimic_{task}_{split}.tsv','w', newline='')
            wr = csv.writer(f, delimiter='\t')
            for content in contents:
                wr.writerow(content)
            f.close()

def process_image(image_path):
    """
    Processes a single image, converting it to Base64.
    """
    try:
        img = Image.open(image_path)
        img_buffer = BytesIO()
        img.save(img_buffer, format=img.format)
        byte_data = img_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode("utf-8")
        #logging.info(f"Successfully processed: {image_path}")
        return base64_str  # You can choose to save/use this result
    except FileNotFoundError:
        logging.error(f"File not found: {image_path}")
    except IOError as e:
        logging.error(f"I/O error for {image_path}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error for {image_path}: {e}")

if __name__ == "__main__":
    main()