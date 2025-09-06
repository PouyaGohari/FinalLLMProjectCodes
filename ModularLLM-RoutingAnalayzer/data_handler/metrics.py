import evaluate
import numpy as np
import logging


loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    if "transformers" in logger.name.lower():
        logger.setLevel(logging.ERROR)


bleu_metric = evaluate.load('bleu', trust_remote_code=True)
rouge_metric = evaluate.load('rouge', trust_remote_code=True)


def compute_generation_metrics(labels, predictions):

    bleu_output = bleu_metric.compute(
        predictions=predictions, references=labels, max_order=1)['bleu']
    rouge_output = rouge_metric.compute(
        predictions=predictions, references=labels, rouge_types=['rougeL'])['rougeL']

    return {
        'bleu': bleu_output,
        'rouge': rouge_output,
    }