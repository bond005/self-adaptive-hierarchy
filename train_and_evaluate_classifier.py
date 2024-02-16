from argparse import ArgumentParser
import os
import random
from typing import Dict
import warnings

from datasets import load_dataset, load_from_disk
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, DataCollatorWithPadding

from neural_network.hierarchical_bert import DistilBertForHierarchicalSequenceClassification


RANDOM_SEED: int = 42
BERT_NAME = 'distilbert/distilbert-base-cased'
ID_TO_LABEL: Dict[str, Dict[int, str]] = {
    'sst2': {0: 'negative', 1: 'positive'},
    'mnli': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'mrpc': {0: 'not_equivalent', 1: 'equivalent'}
}
tokenizer: DistilBertTokenizer
number_of_classes: int


def mnli_to_array_fn(batch):
    premise = batch['premise']
    hypothesis = batch['hypothesis']
    input_ids = tokenizer.encode(
        tokenizer.cls_token + premise + tokenizer.sep_token + hypothesis + tokenizer.sep_token,
        add_special_tokens=False
    )
    batch['input_ids'] = input_ids
    return batch


def mrpc_to_array_fn(batch):
    sentence1 = batch['sentence1']
    sentence2 = batch['sentence2']
    input_ids = tokenizer.encode(
        tokenizer.cls_token + sentence1 + tokenizer.sep_token + sentence2 + tokenizer.sep_token,
        add_special_tokens=False
    )
    batch['input_ids'] = input_ids
    return batch


def sst2_to_array_fn(batch):
    sentence = batch['sentence']
    input_ids = tokenizer.encode(
        sentence,
        add_special_tokens=True
    )
    batch['input_ids'] = input_ids
    return batch


def compute_metrics(eval_pred):
    probabilities, labels = eval_pred
    y_true = labels.flatten().astype(np.int32)
    y_pred = np.argmax(probabilities, axis=-1).flatten().astype(np.int32)
    res = {
        'f1': f1_score(y_true=y_true, y_pred=y_pred, average='binary' if (number_of_classes < 3) else 'macro'),
        'accuracy': accuracy_score(y_true=y_true, y_pred=y_pred)
    }
    del y_true, y_pred
    return res


def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_model_name', type=str, required=False,
                        default=BERT_NAME, help='The input pre-trained BERT name.')
    parser.add_argument('-o', '--output', dest='output_model_name', type=str, required=False,
                        default=BERT_NAME, help='The output BERT name after its fine-tuning.')
    parser.add_argument('-t', '--task', dest='task_name', type=str, required=True,
                        choices=['sst2', 'mnli', 'mrpc'], help='The solved task in the GLUE benchmark.')
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
    parser.add_argument('--lr', dest='learning_rate', type=float, required=False, default=1e-5,
                        help='The learning rate.')
    parser.add_argument('--patience', dest='patience', type=int, required=False, default=3,
                        help='The patience for the early stopping.')
    parser.add_argument('--stepsize', dest='step_size', type=int, required=False, default=None,
                        help='The step size, i.e. samples per epoch.')
    args = parser.parse_args()

    n_processes = max(1, os.cpu_count())
    print(f'Number of CPU cores is {n_processes}.')

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    if not torch.cuda.is_available():
        raise ValueError('CUDA is not available!')
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

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
    train_test_split = dataset.train_test_split(shuffle=True, test_size=0.1, seed=args.random_seed)
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
    tokenizer = DistilBertTokenizer.from_pretrained(args.input_model_name)
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

    id2label = ID_TO_LABEL[args.task_name]
    label2id = dict([(ID_TO_LABEL[args.task_name][label_id], label_id) for label_id in set_of_class_labels])
    if args.use_hierarchy:
        model = DistilBertForHierarchicalSequenceClassification.from_pretrained(
            args.input_model_name,
            num_labels=number_of_classes, id2label=id2label, label2id=label2id
        ).cuda()
    else:
        model = DistilBertForSequenceClassification.from_pretrained(
            args.input_model_name,
            num_labels=number_of_classes, id2label=id2label, label2id=label2id
        ).cuda()
    model.eval()
    print(f'The model is loaded from the {args.input_model_name}.')

    if args.step_size is None:
        iters_per_epoch = 1
        strategy = 'epoch'
    else:
        iters_per_epoch = max(args.step_size // args.minibatch, 3)
        strategy = 'steps'
        print(f'Iterations per epoch is {iters_per_epoch}.')
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=output_model_name,
        overwrite_output_dir=True,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.minibatch,
        per_device_eval_batch_size=args.minibatch,
        num_train_epochs=args.epochs_number,
        evaluation_strategy=strategy,
        eval_steps=iters_per_epoch,
        save_strategy=strategy,
        save_steps=iters_per_epoch,
        save_total_limit=2,
        save_safetensors=False,
        logging_strategy=strategy,
        logging_steps=iters_per_epoch,
        load_best_model_at_end=True,
        metric_for_best_model='eval_f1',
        greater_is_better=True,
        gradient_checkpointing=False,
        seed=args.random_seed,
        data_seed=args.random_seed
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_set,
        eval_dataset=validation_set,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )
    trainer.train()
    model.save_pretrained(output_model_name, safe_serialization=False)
    model.save_pretrained(output_model_name, safe_serialization=True)

    predictions, label_ids, metrics = trainer.predict(test_set)
    metric_name_width = max([len(metric_name) for metric_name in metrics])
    print('Test results:')
    for metric_name in metrics:
        print('  - {0:<{1}} = {2:.6f}'.format(metric_name, metric_name_width, metrics[metric_name]))
    print('')
    target_names = [id2label[label_id] for label_id in sorted(list(id2label.keys()))]
    print(classification_report(y_true=label_ids, y_pred=predictions, target_names=target_names, digits=4))


if __name__ == '__main__':
    main()
