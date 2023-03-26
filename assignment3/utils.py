import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets
import pandas as pd
import os
import collections
from collections import Counter
import torch.nn.utils.prune as prune
from torch import nn

def remove_duplicates(raw_dataset):
    """
    Removing duplicates from HuggingFace dataset. Has only 'train' for LAMA
    Remove if the sentences are same present in masked_sentences. 
    Note: uuid is same but the sentences are different. So, not using that
    """
    dataset_dict = DatasetDict()
    df_train = raw_dataset['train'].to_pandas()
    # print(f"Dataset shape before removing duplicates: {df_train.shape}")
    df_train = df_train.drop_duplicates(subset=['masked_sentence'])
    # print(f"Dataset shape after removing duplicates: {df_train.shape}")
    dataset = Dataset.from_pandas(df_train)
    dataset_dict['train'] = dataset
    return dataset_dict

def list_to_str(example):
    example['masked_sentence'] = example['masked_sentences'][0]
    return example

def extract_dataset(dataset_name):
    if dataset_name == 'conceptnet':
        dataset = load_dataset('json', data_files='bertnesia_data/conceptnet.json')
        dataset = dataset.map(list_to_str)
    if dataset_name == 'squad':
        dataset = load_dataset('json', data_files='bertnesia_data/squad.json')
        dataset = dataset.map(list_to_str)
    if dataset_name == 'trex':
        dataset = load_dataset('json', data_files='bertnesia_data/trex.json')
        dataset = dataset.map(list_to_str)
    if dataset_name == 'google_re':
        dataset = load_dataset('json', data_files='bertnesia_data/google_re.json')
        dataset = dataset.map(list_to_str)

    # Don't use batched=True as we are indexing the list and it's throwing error!
    # print(dataset)
    return dataset


def tokenizer_debug(tokenize_dataset, tokenizer):
    # Sampled dataset for debugging
    small_dataset = tokenize_dataset['train'].select(range(128))
    # Just to verify whether entire thing is going well
    detokenised_text = []
    for inp in small_dataset:
        tokens = tokenizer.decode(inp['input_ids'])
        label_tokens = tokenizer.decode(inp['output_labels'])
        detokenised_text.append((tokens, label_tokens))
    count = 1
    for inp, label in detokenised_text:
        print(count)
        print(f"Input: {inp}")
        print(f"Label: {label}")
        print("------")
        count += 1

def dataloader_debug(train_dataloader):
    i = 0
    for batch in train_dataloader:
        print(batch['input_ids'].shape)
        print("**************")
        print(batch['output_labels'].shape)
        if i == 3:
            break
        else:
            print("---------")
        i += 1

def evaluate(batch_labels, batch_outputs):
    top_5_correct = 0
    top_1_correct = 0
    # Uncomment if you want to debug/verify
    # This lowercase is needed for roberta-base as it is case sensitive model
    # bert already is case insensitive, so it won't be an issue imo
    batch_outputs = [[x.lower() for x in y] for y in batch_outputs]
    # print("----------------")
    for i in range(len(batch_outputs)):
        # print(f"Output: {batch_outputs[i]}")
        # print(f"Labels: {batch_labels[i]}")
        if batch_labels[i] in batch_outputs[i]:
            top_5_correct += 1
        if batch_labels[i] == batch_outputs[i][0]:
            top_1_correct += 1
    return top_1_correct, top_5_correct

def inference(model, tokenizer, device, train_dataloader, dataset_name, prune_type, prune_percentage, layer_index):
    total_correct_top_1, total_correct_top_5, total_count = 0, 0, 0
    model.eval()
    for batch in train_dataloader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['output_labels'].to(device)

            batch_inputs = [tokenizer.decode(inputs) for inputs in input_ids]
            batch_labels = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]

            token_logits = model(input_ids, attention_mask=attention_mask).logits
            # print(f"Token logits: {token_logits.shape}")

            mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)
            # print(mask_token_index)

            # We have to index with the input sentence as the first dimension and the location of MASK in 2nd dimension
            # token_logits[:, mask_token_index[1], :] is incorrect when checking the shapes
            mask_token_logits = token_logits[mask_token_index[0], mask_token_index[1], :]
            # print(f"Mask token logits: {mask_token_logits.shape}")

            # Remove the batch labels for which there's no [MASK], apparently some examples don't have [MASK] 
            batch_labels = [batch_labels[i] for i in mask_token_index[0]]

            top_5_tokens = torch.topk(mask_token_logits, 5, dim=-1).indices
            top_5_tokens.to(device)

            # Nested for loop in list comprehension, to get a list of lists where the inner list has the predictions for a particular
            # sentence 
            batch_outputs = [[tokenizer.decode(predicted_token) for predicted_token in predicted] for predicted in top_5_tokens]
            
            # Uncomment if needed
            # for inp, label, output in zip(batch_inputs, batch_labels, batch_outputs):
            #     print(f"Input: {inp}")
            #     print(f"Label: {label}")
            #     print(f"Output: {output}")
            #     print("\n")

            top_1_correct, top_5_correct = evaluate(batch_labels, batch_outputs)
            total_correct_top_1 += top_1_correct
            total_correct_top_5 += top_5_correct
            total_count += len(batch_labels)
            # Uncomment if needed
            # print(f"Top 1, top 5 and total in this batch is {total_correct_top_1}, {total_correct_top_5}, {total_count}")
            # print("**********")

    print(f"Experiment name: {dataset_name}, {prune_type}, {prune_percentage}, Layer number: {layer_index}")
    print(f"Top 1, 5 and total labels are: {total_correct_top_1}, {total_correct_top_5}, {total_count}")
    accuracy_top_5 = round(100*(total_correct_top_5 / total_count), 3)
    accuracy_top_1 = round(100*(total_correct_top_1 / total_count), 3)
    print(f"Top 1 Accuracy: {accuracy_top_1}, Top 5 Accuracy: {accuracy_top_5}")
    print("----------------------------")
    return 0

def local_pruning(model, linear_layers_list, layer_index, prune_percentage, prune_type, n=1):
    """
    We use n only for ln_structured. For other prunings, n is not needed or n=1
    Todos: 
        1. Currently works only if weight is there for layer
        2. dim. Which one we need to prune!
        3. Global pruning
    """
    layer = linear_layers_list[layer_index]
    print(f"Layer name is {layer}")
    if prune_type == 'l1_unstructured':
        # L1 unstructured
        prune.l1_unstructured(layer, name="weight", amount=prune_percentage)

    if prune_type == 'random_unstructured':
        # Random unstructured
        prune.random_unstructured(layer, name='weight', amount=prune_percentage)

    if prune_type == 'random_structured':
        # Random structured (Didn't understand role of dim, [768, 30522] which one we should prune!)
        prune.random_structured(layer, name='weight', amount=prune_percentage, dim=0)
    
    if prune_type == 'ln_structured':
        # Ln structured with n=1 i.e L1 pruning
        prune.ln_structured(layer, name='weight', amount=prune_percentage, dim=0, n=n)
    
    print(f"Percentage pruned is : {100* torch.sum(layer.weight == 0)/layer.weight.nelement()}")
    print(f"List of pruned layers: {dict(model.named_buffers()).keys()}")
    print(f"Pruning type: {layer._forward_pre_hooks}")
    # This state dict will give all the layer weights related details to crosscheck
    # print(f"Pruned state dicts: {model.state_dict().keys()}")
    parameters_to_prune = tuple((x, 'weight') for x in linear_layers_list)
    get_global_sparsity(parameters_to_prune)

def get_total_parameters(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total and trainable parameters are : {pytorch_total_params}, {pytorch_trainable_params}")

def get_global_sparsity(parameters_to_prune):
    sparsed_weights, total_weights = 0.0, 0.0
    for layer in parameters_to_prune:
        # Uncomment to see each layer's sparsity!
        # print(f"Percentage pruned is : {100* torch.sum(layer[0].weight == 0)/layer[0].weight.nelement()}")
        # print(f"List of pruned layers: {dict(layer[0].named_buffers()).keys()}")
        # print(f"Pruning type: {layer[0]._forward_pre_hooks}")

        sparsed_weights += torch.sum(layer[0].weight == 0)
        total_weights += layer[0].weight.nelement()

    print(f"Global sparsity is: {100*(sparsed_weights/total_weights)}")
    
def global_pruning(linear_layers_list, prune_percentage):
    """
    Global pruning takes sometime to execute!
    """
    parameters_to_prune = tuple((x, 'weight') for x in linear_layers_list)
    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=prune_percentage)
    get_global_sparsity(parameters_to_prune)

def instantiate_model(model, model_layers_list):
    bert_output_layers = []
    linear_layers_list = []
    
    for layer_name in model.modules():
        if layer_name.__class__.__name__ in model_layers_list:
            bert_output_layers.append(layer_name)
    for bert_output in bert_output_layers:
        for layer_name in bert_output.modules():
            if isinstance(layer_name, nn.Linear):
                linear_layers_list.append(layer_name)
    # print("----------------------------")
    # print(linear_layers_list)
    # print(f"No of linear layers are: {len(linear_layers_list)}")
    return linear_layers_list