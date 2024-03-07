from argparse import ArgumentParser
import re
import gc
import os
import random
from typing import Dict, Set, Union
import warnings

from datasets import load_dataset
from datasets import Dataset
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
import torch
from transformers import get_constant_schedule_with_warmup, get_constant_schedule
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, DataCollatorWithPadding

from neural_network.hierarchical_distilbert import DistilBertForHierarchicalSequenceClassification
from neural_network.hierarchical_bert import BertForHierarchicalSequenceClassification
from neural_network.utils import HierarchyPrinterCallback


RANDOM_SEED: int = 42
BERT_NAME = 'distilbert/distilbert-base-cased'
ID_TO_LABEL: Dict[str, Dict[int, str]] = {
    'sst2': {0: 'negative', 1: 'positive'},
    'mnli': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'mrpc': {0: 'not_equivalent', 1: 'equivalent'}
}
input_prompt: Union[str, None] = None
tokenizer: Union[DistilBertTokenizer, BertTokenizer]
number_of_classes: int


def check_prompt(prompt: str) -> int:
    re_for_check = re.compile(r'{\d+}')
    found_idx = prompt.find('{0}')
    if found_idx < 0:
        if re_for_check.search(prompt) is not None:
            raise ValueError(f'The prompt "{prompt}" is incorrect!')
        n_fields = 0
    else:
        found_idx_ = prompt[(found_idx + 3):].find('{1}')
        if found_idx_ < 0:
            if re_for_check.search(prompt[(found_idx + 3):]) is not None:
                raise ValueError(f'The prompt "{prompt}" is incorrect!')
            n_fields = 1
        else:
            found_idx += (found_idx_ + 3)
            if re_for_check.search(prompt[(found_idx + 3):]) is not None:
                raise ValueError(f'The prompt "{prompt}" is incorrect!')
            n_fields = 2
    return n_fields


def mnli_to_array_fn(batch):
    premise = batch['premise'].strip()
    hypothesis = batch['hypothesis'].strip()
    if input_prompt is None:
        input_ids = tokenizer.encode(
            tokenizer.cls_token + premise + tokenizer.sep_token + hypothesis + tokenizer.sep_token,
            add_special_tokens=False
        )
    else:
        united_sentence = input_prompt.format(premise, hypothesis)
        input_ids = tokenizer.encode(
            united_sentence,
            add_special_tokens=True
        )
    batch['input_ids'] = input_ids
    return batch


def mrpc_to_array_fn(batch):
    sentence1 = batch['sentence1'].strip()
    sentence2 = batch['sentence2'].strip()
    if input_prompt is None:
        input_ids = tokenizer.encode(
            tokenizer.cls_token + sentence1 + tokenizer.sep_token + sentence2 + tokenizer.sep_token,
            add_special_tokens=False
        )
    else:
        united_sentence = input_prompt.format(sentence1, sentence2)
        input_ids = tokenizer.encode(
            united_sentence,
            add_special_tokens=True
        )
    batch['input_ids'] = input_ids
    return batch


def sst2_to_array_fn(batch):
    sentence = batch['sentence'].strip()
    if input_prompt is not None:
        sentence = input_prompt.format(sentence)
    input_ids = tokenizer.encode(
        sentence,
        add_special_tokens=True
    )
    batch['input_ids'] = input_ids
    return batch


def compute_metrics(eval_pred):
    probabilities, labels = eval_pred
    y_true = labels.flatten().astype(np.int32)
    if isinstance(probabilities, np.ndarray):
        y_pred = np.argmax(probabilities, axis=-1).flatten().astype(np.int32)
    else:
        y_pred = np.argmax(probabilities[0], axis=-1).flatten().astype(np.int32)
    res = {
        'f1': f1_score(y_true=y_true, y_pred=y_pred, average='binary' if (number_of_classes < 3) else 'macro'),
        'accuracy': accuracy_score(y_true=y_true, y_pred=y_pred)
    }
    del y_true, y_pred
    return res


def train(task_name: str, set_of_class_labels: Set[int], model_type: str,
          training_set: Dataset, validation_set: Dataset, test_set: Dataset,
          input_model_name: str, output_model_name: str, metric_name: str,
          use_hierarchy: bool, freeze_hierarchy: bool, freeze_transformer: bool,
          max_epochs: int, patience: Union[int, None], minibatch: int, learning_rate: float, warmup: int,
          random_seed: int):
    if model_type not in {'bert', 'distilbert'}:
        raise ValueError(f'The model type {model_type} is not support!')
    print('')
    id2label = ID_TO_LABEL[task_name]
    label2id = dict([(ID_TO_LABEL[task_name][label_id], label_id) for label_id in set_of_class_labels])
    if use_hierarchy:
        if model_type == 'distilbert':
            model = DistilBertForHierarchicalSequenceClassification.from_pretrained(
                input_model_name,
                num_labels=number_of_classes, id2label=id2label, label2id=label2id
            ).cuda()
            print('The DistilBERT architecture is used.')
        else:
            model = BertForHierarchicalSequenceClassification.from_pretrained(
                input_model_name,
                num_labels=number_of_classes, id2label=id2label, label2id=label2id
            ).cuda()
            print('The BERT architecture is used.')
        print(f'Initial layer importances are: {model.layer_importances}.')
        if freeze_hierarchy:
            for param in model.layer_weights.parameters():
                param.requires_grad = False
    else:
        if model_type == 'distilbert':
            model = DistilBertForSequenceClassification.from_pretrained(
                input_model_name,
                num_labels=number_of_classes, id2label=id2label, label2id=label2id
            ).cuda()
            print('The DistilBERT architecture is used.')
        else:
            model = BertForSequenceClassification.from_pretrained(
                input_model_name,
                num_labels=number_of_classes, id2label=id2label, label2id=label2id
            ).cuda()
            print('The BERT architecture is used.')
    model.eval()
    if freeze_transformer:
        if model_type == 'distilbert':
            for param in model.distilbert.parameters():
                param.requires_grad = False
        else:
            for param in model.bert.parameters():
                param.requires_grad = False
        print(f'The model with frozen base (transformer) is loaded from the {input_model_name}.')
    else:
        print(f'The model is loaded from the {input_model_name}.')

    print('')
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=output_model_name,
        overwrite_output_dir=True,
        learning_rate=learning_rate,
        per_device_train_batch_size=minibatch,
        per_device_eval_batch_size=minibatch,
        num_train_epochs=max_epochs,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        save_safetensors=False,
        logging_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model=f'eval_{metric_name}',
        greater_is_better=(metric_name != 'loss'),
        gradient_checkpointing=False,
        seed=random_seed,
        data_seed=random_seed
    )
    if patience is None:
        callbacks = []
    else:
        callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]
    if use_hierarchy:
        callbacks.append(HierarchyPrinterCallback(num_layers=5))
    steps_per_epoch = max(1, len(training_set) // minibatch)
    print(f'Number of steps per epoch is {steps_per_epoch}.')
    num_training_steps = steps_per_epoch * max_epochs
    print(f'Total number of training steps is {num_training_steps}.')
    if warmup > 0:
        num_warmup_steps = steps_per_epoch * warmup
        print(f'Number of warmup steps is {num_warmup_steps}.')
    else:
        num_warmup_steps = 0
    optim = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    if warmup > 0:
        scheduler = get_constant_schedule_with_warmup(optimizer=optim, num_warmup_steps=num_warmup_steps)
    else:
        scheduler = get_constant_schedule(optimizer=optim)
    if len(callbacks) > 0:
        trainer = Trainer(
            model=model,
            args=training_args,
            optimizers=(optim, scheduler),
            train_dataset=training_set,
            eval_dataset=validation_set,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            optimizers=(optim, scheduler),
            train_dataset=training_set,
            eval_dataset=validation_set,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )
    trainer.train()
    model.save_pretrained(output_model_name, safe_serialization=False)
    model.save_pretrained(output_model_name, safe_serialization=True)

    predictions, label_ids, metrics = trainer.predict(test_set)
    metric_name_width = max([len(metric_name) for metric_name in metrics])
    print('')
    if freeze_transformer:
        print('Test results (with frozen base):')
    else:
        print('Test results:')
    for metric_name in metrics:
        print('  - {0:<{1}} = {2:.6f}'.format(metric_name, metric_name_width, metrics[metric_name]))
    print('')
    target_names = [id2label[label_id] for label_id in sorted(list(id2label.keys()))]
    if isinstance(predictions, np.ndarray):
        y_pred = np.argmax(predictions, axis=-1)
    else:
        y_pred = np.argmax(predictions[0], axis=-1)
    print(classification_report(y_true=label_ids, y_pred=y_pred, target_names=target_names, digits=4))
    print('')


def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_model_name', type=str, required=False,
                        default=BERT_NAME, help='The input pre-trained BERT name.')
    parser.add_argument('-o', '--output', dest='output_model_name', type=str, required=False,
                        default=BERT_NAME, help='The output BERT name after its fine-tuning.')
    parser.add_argument('-t', '--task', dest='task_name', type=str, required=True,
                        choices=['sst2', 'mnli', 'mrpc'], help='The solved task in the GLUE benchmark.')
    parser.add_argument('-m', '--model', dest='model_type', type=str, required=True,
                        choices=['bert', 'distilbert'], help='The model type (BERT or DistilBERT).')
    parser.add_argument('-d', '--dataset', dest='dataset_name', type=str, required=False, default='glue',
                        help='The path to the GLUE benchmark.')
    parser.add_argument('-r', '--random', dest='random_seed', type=int, required=False, default=42,
                        help='The random seed.')
    parser.add_argument('--hierarchy', dest='use_hierarchy', action='store_true',
                        help='Is the self-adaptive hierarchical BERT used instead of the usual BERT?')
    parser.add_argument('--minibatch', dest='minibatch', type=int, required=False, default=64,
                        help='The minibatch size.')
    parser.add_argument('--epochs', dest='epochs_number', type=int, required=False, default=100,
                        help='The maximal number of epochs.')
    parser.add_argument('--lr1', dest='learning_rate_1', type=float, required=False, default=1e-3,
                        help='The learning rate on the first phase (fine-tuning with frozen Transformer).')
    parser.add_argument('--lr2', dest='learning_rate_2', type=float, required=False, default=1e-6,
                        help='The learning rate on the second phase (fine-tuning with unfrozen Transformer).')
    parser.add_argument('--patience', dest='patience', type=int, required=False, default=None,
                        help='The patience for the early stopping.')
    parser.add_argument('--warmup', dest='warmup', type=int, required=False, default=3,
                        help='The number of warmup steps to do.')
    parser.add_argument('--metric', dest='metric_name', type=str, required=True,
                        choices=['loss', 'accuracy', 'f1'], help='The monitored metric for early stopping.')
    parser.add_argument('--prompt', dest='prompt', type=str, required=False, default=None,
                        help='The additional input prompt.')
    args = parser.parse_args()

    n_processes = max(1, os.cpu_count())
    print(f'Number of CPU cores is {n_processes}.')

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    if not torch.cuda.is_available():
        raise ValueError('CUDA is not available!')
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    global input_prompt
    if args.prompt is None:
        input_prompt = None
    else:
        input_prompt = args.prompt.strip()
        if len(input_prompt) > 0:
            input_prompt = ' '.join(input_prompt.split())
            number_of_fields = check_prompt(input_prompt)
            if number_of_fields > 0:
                if args.task_name == 'sst2':
                    if number_of_fields != 1:
                        raise ValueError(f'The prompt "{input_prompt}" is incorrect!')
                else:
                    if number_of_fields != 2:
                        raise ValueError(f'The prompt "{input_prompt}" is incorrect!')
        else:
            input_prompt = None

    output_model_name = os.path.normpath(args.output_model_name)
    if len(output_model_name) == 0:
        raise ValueError(f'The output BERT name after its fine-tuning is empty!')
    if not os.path.isdir(output_model_name):
        basedir = os.path.dirname(output_model_name)
        if len(basedir) > 0:
            if not os.path.isdir(basedir):
                raise ValueError(f'The directory "{basedir}" does not exist!')

    if os.path.isdir(args.dataset_name):
        dataset = load_dataset(os.path.join(str(args.dataset_name), str(args.task_name)), split='train',
                               num_proc=n_processes)
    else:
        dataset = load_dataset(args.dataset_name, args.task_name, split='train', num_proc=n_processes)
    train_test_split = dataset.train_test_split(shuffle=True, test_size=0.3, seed=args.random_seed,
                                                stratify_by_column='label')
    training_set = train_test_split['train']
    validation_set = train_test_split['test']
    del dataset, train_test_split
    if os.path.isdir(args.dataset_name):
        test_set = load_dataset(os.path.join(str(args.dataset_name), str(args.task_name)), split='validation',
                                num_proc=n_processes)
    else:
        test_set = load_dataset(args.dataset_name, args.task_name, split='validation', num_proc=n_processes)
    print(f'The dataset {os.path.basename(args.dataset_name)}[{args.task_name}] is loaded.')
    print(f'Number of training samples is {len(training_set)}.')
    print(f'Number of validation samples is {len(validation_set)}.')
    print(f'Number of test samples is {len(test_set)}.')

    global tokenizer
    if args.model_type == 'distilbert':
        tokenizer = DistilBertTokenizer.from_pretrained(args.input_model_name)
    else:
        tokenizer = BertTokenizer.from_pretrained(args.input_model_name)
    print(f'The BERT tokenizer is loaded from "{args.input_model_name}".')

    removed_columns = list(set(training_set.column_names) - {'label', 'input_ids'})
    if args.task_name == 'mnli':
        process_fn = mnli_to_array_fn
    elif args.task_name == 'mrpc':
        process_fn = mrpc_to_array_fn
    else:
        process_fn = sst2_to_array_fn
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        training_set = training_set.map(
            process_fn,
            remove_columns=removed_columns,
            num_proc=n_processes
        )
        validation_set = validation_set.map(
            process_fn,
            remove_columns=removed_columns,
            num_proc=n_processes
        )
        test_set = test_set.map(
            process_fn,
            remove_columns=removed_columns,
            num_proc=n_processes
        )
    print(f'The dataset {os.path.basename(args.dataset_name)}[{args.task_name}] is tokenized.')
    print(f'Fields of the training set are: {training_set.column_names}.')
    print(f'Fields of the validation set are: {validation_set.column_names}.')
    print(f'Fields of the test set are: {test_set.column_names}.')

    set_of_class_labels = set()
    for sample in training_set['label']:
        set_of_class_labels.add(int(sample))
    if set_of_class_labels != set(ID_TO_LABEL[args.task_name].keys()):
        err_msg = f'The label set {set_of_class_labels} is wrong for the task {args.task_name}.'
        raise ValueError(err_msg)
    for sample in validation_set['label']:
        if int(sample) not in set_of_class_labels:
            err_msg = f'The label {sample} is unknown for the task {args.task_name}.'
            raise ValueError(err_msg)
    for sample in test_set['label']:
        if int(sample) not in set_of_class_labels:
            err_msg = f'The label {sample} is unknown for the task {args.task_name}.'
            raise ValueError(err_msg)
    global number_of_classes
    number_of_classes = len(set_of_class_labels)
    print(f'The number of classes is {number_of_classes}.')
    print(f'They are: {[ID_TO_LABEL[args.task_name][label_id] for label_id in sorted(list(set_of_class_labels))]}')

    print('')
    print('Three random samples from the training data are:')
    print('')
    for sample_idx in random.sample(list(range(len(training_set))), k=3):
        input_ids = training_set['input_ids'][sample_idx]
        print(input_ids)
        print(tokenizer.decode(input_ids, skip_special_tokens=False))
        print('')

    train(task_name=args.task_name, set_of_class_labels=set_of_class_labels,
          training_set=training_set, validation_set=validation_set, test_set=test_set,
          input_model_name=args.input_model_name, output_model_name=args.output_model_name, model_type=args.model_type,
          use_hierarchy=args.use_hierarchy, freeze_transformer=True, freeze_hierarchy=False,
          max_epochs=args.epochs_number, warmup=0, patience=args.patience, minibatch=args.minibatch,
          learning_rate=args.learning_rate_1, random_seed=args.random_seed, metric_name=args.metric_name)

    torch.cuda.empty_cache()
    gc.collect()

    train(task_name=args.task_name, set_of_class_labels=set_of_class_labels,
          training_set=training_set, validation_set=validation_set, test_set=test_set,
          input_model_name=args.output_model_name, output_model_name=args.output_model_name, model_type=args.model_type,
          use_hierarchy=args.use_hierarchy, freeze_transformer=False, freeze_hierarchy=True,
          max_epochs=args.epochs_number, warmup=args.warmup, patience=args.patience, metric_name=args.metric_name,
          minibatch=max(args.minibatch // 4, 1), learning_rate=args.learning_rate_2, random_seed=args.random_seed)


if __name__ == '__main__':
    main()
