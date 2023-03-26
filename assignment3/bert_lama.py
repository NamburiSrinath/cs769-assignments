"""
BERT with LAMA Probing

Todos:
1. Dynamic padding to speeden up!!
2. Evaluation loop to avoid decoding steps and directly comparing in token id space!!
3. Few examples don't have <mask>, especially in trex, google re. For now, skipping it
4. Few examples have multiple <mask>. But the actual label is just one when printing!
"""
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from utils import remove_duplicates, extract_dataset, inference, local_pruning, instantiate_model, global_pruning
import torch.nn.utils.prune as prune
from torch import nn
from transformers.utils import logging
logging.set_verbosity(40)

def tokenize_function(example):
    tokenized_text = tokenizer(example['masked_sentence'], truncation=True,
                                padding='max_length', max_length=tokenizer.model_max_length)
    tokenized_labels = tokenizer(example['obj_label'], truncation=True, padding='max_length', max_length=8)
    tokenized_data = {
        "input_ids": tokenized_text['input_ids'],
        "attention_mask": tokenized_text['attention_mask'],
        "token_type_ids": tokenized_text['token_type_ids'],
        "output_labels": tokenized_labels['input_ids']
    }
    return tokenized_data


if __name__ == '__main__':
    # dataset_name = One among 'trex', 'conceptnet', 'google_re', 'squad'
    # For quicker experiments, try running squad, conceptnet first, then google_re and trex at last
    dataset_name_list = ['squad', 'conceptnet', 'trex', 'google_re']
    prune_layers_list = ['BertOutput', 'BertOnlyMLMHead']
    checkpoint = 'bert-base-uncased'
    batch_size=196

    for dataset_name in dataset_name_list:
        # Extract the preprocessed dataset with BERTnesia codebase 
        raw_dataset = extract_dataset(dataset_name)
        # print(raw_dataset)
        
        # Loading from HF is fine with Conceptnet and Squad but not for TREx and Google_RE
        # raw_dataset = load_dataset('lama', dataset_name)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        print(f"Fast tokenizer is available: {tokenizer.is_fast}")
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Tokenize the dataset
        tokenize_dataset = raw_dataset.map(tokenize_function, batched=True)
        # print(tokenize_dataset['train'].column_names)

        # Remove the duplicates
        tokenize_dataset = remove_duplicates(tokenize_dataset)
        print(tokenize_dataset)
        
        # Remove columns and set it to Pytorch format
        tokenize_dataset = tokenize_dataset.remove_columns([col for col in tokenize_dataset['train'].column_names
                                            if col not in ['input_ids', 'attention_mask', 'output_labels', 'token_type_ids']])
        tokenize_dataset.set_format(type='torch')
        # Uncomment if needed, this decodes the tokenized dataset and prints it
        # tokenizer_debug(tokenize_dataset, tokenizer)

        # Dataloader with shuffle true
        train_dataloader = DataLoader(tokenize_dataset['train'], batch_size=batch_size, shuffle=True, collate_fn=data_collator)
        # Uncomment if needed, this prints the datashapes 
        # dataloader_debug(train_dataloader)

        
        # last_decoder = model.cls.predictions.decoder
        prune_percentage_list = [0, 0.2, 0.4]
        prune_type_list = ['l1_unstructured', 'random_unstructured', 'random_structured', 'ln_structured', 'global_pruning', 'baseline']
        for prune_percentage in prune_percentage_list:
            for prune_type in prune_type_list:
                model = AutoModelForMaskedLM.from_pretrained(checkpoint)
                linear_layers_list = instantiate_model(model, prune_layers_list)
                if prune_percentage == 0 and prune_type == 'baseline':
                    model = AutoModelForMaskedLM.from_pretrained(checkpoint)
                    model.to(device)
                    inference(model, tokenizer, device, train_dataloader, dataset_name, prune_type, prune_percentage, -1)
                if prune_percentage != 0:
                    if prune_type == 'global_pruning':
                        model = AutoModelForMaskedLM.from_pretrained(checkpoint)
                        linear_layers_list = instantiate_model(model, prune_layers_list)
                        # Global pruning
                        global_pruning(linear_layers_list, prune_percentage=prune_percentage)
                        model.to(device)
                        inference(model, tokenizer, device, train_dataloader, dataset_name, prune_type, prune_percentage, -1)
                    else:
                        for layer_index in range(len(linear_layers_list)):
                            # Incase we want some stats on no of parameters
                            # get_total_parameters(model)
                            model = AutoModelForMaskedLM.from_pretrained(checkpoint)
                            linear_layers_list = instantiate_model(model, prune_layers_list)
                            # Local pruning 
                            local_pruning(model, linear_layers_list, layer_index, prune_percentage=prune_percentage, prune_type=prune_type,n=1)
                            model.to(device)
                            inference(model, tokenizer, device, train_dataloader, dataset_name, prune_type, prune_percentage, layer_index)
