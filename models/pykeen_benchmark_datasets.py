#!/usr/bin/python3

import wandb
import os
import torch
from pykeen.pipeline import pipeline
import pandas as pd

os.environ["WANDB_API_KEY"] = "a0dcca4cf18920b5c23ec09023f46ffa76caad5b"
wandb.login()

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run(params):
    pipeline_result = pipeline(
        model=params['model'],
        dataset=params['dataset'],
        model_kwargs=dict(
        embedding_dim=params['embedding_dim'],
        ),
        training_kwargs=dict(
            num_epochs=80000,
            batch_size=params['batch_size'],
        ),
        device=dev,
        loss=params['loss'],
        optimizer='Adam',
        optimizer_kwargs=dict(
            lr=params['lr'],
        ),
        negative_sampler_kwargs = dict(
                num_negs_per_pos = params['num_negs_per_pos'],
        ),
        result_tracker='wandb',
        result_tracker_kwargs=dict(
            project='KG-benchmarks',
        ),
        stopper='early',
        stopper_kwargs=dict(frequency=1000, patience=2, relative_delta=0.002),
    )

    mrr = pipeline_result.get_metric('mrr')
    # print("MRR:", mrr)
    hits10 = pipeline_result.get_metric('hits@10')
    # print("Hits@10", hits10)

    return mrr, hits10

# My HPO
# settings = [
#     {
#         'dataset': 'fb15k',
#         'model': 'complex',
#         'loss': 'bcewithlogits',
#         'embedding_dim': 112,
#         'num_negs_per_pos': 3,
#         'lr': 0.08,
#         'batch_size': 512,
#     },
#     {
#         'dataset': 'fb15k',
#         'model': 'rotate',
#         'loss': 'marginranking',
#         'embedding_dim': 272,
#         'num_negs_per_pos': 24,
#         'lr': 0.01,
#         'batch_size': 512,
#     },
#     {
#         'dataset': 'wn18',
#         'model': 'complex',
#         'loss': 'marginranking',
#         'embedding_dim': 192,
#         'num_negs_per_pos': 4,
#         'lr': 0.05,
#         'batch_size': 512,
#     },
#     {
#         'dataset': 'wn18',
#         'model': 'rotate',
#         'loss': 'marginranking',
#         'embedding_dim': 528,
#         'num_negs_per_pos': 1,
#         'lr': 0.05,
#         'batch_size': 512,
#     },
# ]

# From https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/blob/master/best_config.sh
settings = [
#    {
#        'dataset': 'fb15k',
#        'model': 'complex',
#        'loss': 'marginranking',
#        'embedding_dim': 1000,
    #     'num_negs_per_pos': 256,
    #     'lr': 0.001,
    #     'batch_size': 512,
    # },
    # {
    #     'dataset': 'fb15k',
    #     'model': 'rotate',
    #     'loss': 'marginranking',
    #     'embedding_dim': 1000,
    #     'num_negs_per_pos': 256,
    #     'lr': 0.0001,
    #     'batch_size': 1024,
    # },
    # {
    #     'dataset': 'wn18',
    #     'model': 'complex',
    #     'loss': 'marginranking',
    #     'embedding_dim': 500,
    #     'num_negs_per_pos': 1024,
    #     'lr': 0.001,
    #     'batch_size': 512,
    # },
    # {
    #     'dataset': 'wn18',
    #     'model': 'rotate',
    #     'loss': 'marginranking',
    #     'embedding_dim': 500,
    #     'num_negs_per_pos': 1024,
    #     'lr': 0.0001,
    #     'batch_size': 512,
    # },
    # {
    #     'dataset': 'fb15k237',
    #     'model': 'rotate',
    #     'loss': 'marginranking',
    #     'embedding_dim': 1000,
    #     'num_negs_per_pos': 256,
    #     'lr': 0.00005,
    #     'batch_size': 1024,
    # },
    # {
    #     'dataset': 'fb15k237',
    #     'model': 'complex',
    #     'loss': 'marginranking',
    #     'embedding_dim': 1000,
    #     'num_negs_per_pos': 256,
    #     'lr': 0.001,
    #     'batch_size': 512,
    # },
    {
        'dataset': 'wn18rr',
        'model': 'complex',
        'loss': 'marginranking',
        'embedding_dim': 500,
        'num_negs_per_pos': 1024,
        'lr': 0.002,
        'batch_size': 512,
    },
    {
        'dataset': 'wn18',
        'model': 'rotate',
        'loss': 'marginranking',
        'embedding_dim': 500,
        'num_negs_per_pos': 1024,
        'lr': 0.00005,
        'batch_size': 512,
    }
]

df_results = pd.DataFrame(columns=["dataset", "model", "mrr", "hits@10"])
for i, s in enumerate(settings):
    mrr, hits10 = run(s)
    df_results.loc[i] = [s['dataset'], s['model'], mrr, hits10]

print(df_results)
