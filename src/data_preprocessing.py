# Data Preprocessing

# import libraries
import re
from transformers import AutoTokenizer

def data_preprocessing(examples: dict) -> dict:
    '''This function preprocesses the email record by cleaning the email body and subject line.
       This function removes unwanted patterns, attachments, and extra whitespace.

    Args:
        examples (dict): A dictionary containing the email body and subject line.

    Returns:
        dict: The original dictionary with two additional keys for the preprocessed email body and subject line.
    '''

    email_body = examples['clean_email']
    subject = examples['subject_line']

    # remove the << >> patterns in email body
    email_body = re.sub(r'<<.*?>>', '', email_body)
    # remove the lines that contain attachment filenames in email body
    email_body = re.sub(r'[^\n]*\.(doc|docx|xls|xlsx|pdf|ppt|pptx|csv|zip|jpg|jpeg|gif)', '', email_body, flags=re.IGNORECASE)
    # remove extra spaces and tabs in email body
    email_body = re.sub(r'[ \t]+', ' ', email_body).strip()

    # remove extra whitespace in subject
    subject = re.sub(r'\s+', ' ', subject).strip()

    # add the preprocessed email body and subject line to the examples dictionary
    examples['preprocessed_email_body'] = email_body
    examples['preprocessed_subject'] = subject

    return examples


def combine_body_subject(examples: dict) -> dict:
    '''
    This function combines the preprocessed email body and subject line into a single string.
    The combined string is in the format: "Subject: <preprocessed_subject>\nEmail body: <preprocessed_email_body>".

    Args:
        examples (dict): A dictionary containing the preprocessed email body and subject line.

    Returns:
        dict: The original dictionary with an additional key for the combined string.
    '''

    examples['body_subject_combined'] = 'Subject: ' + examples['preprocessed_subject'] +  '\nEmail body:' + examples['preprocessed_email_body']
    
    return examples


def add_instruction(examples: dict, 
                    instruction: str = 'Generate an email subject for the following email:', 
                    email_body_key: str = 'preprocessed_email_body') -> dict:
    '''This function adds an instruction to the email body.

    Args:
        examples (dict): A dictionary containing the email body.
        instruction (str): The instruction to be added.
        email_body_key (str): The key of the email body field that will be combined with the instruction.

    Returns:
        dict: The original dictionary with an additional key for the email bodywith instruction.
    '''
    
    examples['email_with_instruction'] = instruction + '\n' + examples[email_body_key]
    return examples


def prepare_dataset(preprocessed_dataset: dict, 
                    model_name: str,
                    input_key: str = 'preprocessed_email_body',
                    target_key: str = 'preprocessed_subject',
                    max_input_length: int = 256, 
                    max_target_length: int = 32, 
                    truncation: bool = True, 
                    task_type: str = 'continuation'):
    '''
    This function prepares the dataset for training.

    Args:
        preprocessed_dataset (dict): The preprocessed dataset.
        model_name (str): The name of the model.
        input_key (str): The key for the input data.
        target_key (str): The key for the target data.
        max_input_length (int): The maximum length of the input data.
        max_target_length (int): The maximum length of the target data.
        truncation (bool): Whether to truncate the data.
        task_type (str): The type of task.

    Returns:
        tuple: The tokenized dataset and the tokenizer.
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set the pad token to the eos token if it is not already set (for models like distilGPT-2)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # tokenize the dataset
    tokenized_dataset = preprocessed_dataset.map(
        lambda examples: tokenizer(
            examples[input_key], 
            max_length=max_input_length, 
            truncation=truncation),
        batched=True
        )

    if task_type == 'continuation':
        pass
    
    # For subject generation, we need to tokenize the target (subject) as well and add it to the tokenized dataset
    elif task_type == 'subject_generation':
        labels = preprocessed_dataset.map(
        lambda examples: tokenizer(
            examples[target_key], 
            max_length=max_target_length, 
            truncation=truncation),
        batched=True
        )
        # Add the tokenized labels to the tokenized dataset
        for split in labels:
            tokenized_dataset[split] = tokenized_dataset[split].add_column('labels', labels[split]['input_ids'])

    else:
        # If the task_type is not recognized, raise a ValueError
        raise ValueError(f"Invalid task_type: {task_type}. Must be 'continuation' or 'subject_generation'.")
    
    return tokenized_dataset, tokenizer

