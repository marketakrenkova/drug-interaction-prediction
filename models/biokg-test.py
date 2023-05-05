#!/usr/bin/python3

from pykeen.predict import predict_target, predict_triples
from pykeen.pipeline import pipeline
from pykeen.models import TransE, ComplEx
from pykeen.evaluation import RankBasedEvaluator
from pykeen.datasets import get_dataset

import pandas as pd
    
hits_at_10 = []
mrr = []

epochs = list(range(10, 101, 10))

for epoch in epochs:
    
    result = pipeline(
        dataset='BioKG',
        model='ComplEx',
        model_kwargs = dict(
            embedding_dim = 1000
        ),
        optimizer = 'Adam',
        optimizer_kwargs = dict(
            lr = 0.001
        ),
        evaluator=RankBasedEvaluator,
        device='gpu',
        training_kwargs = dict(
                    batch_size = 512,
                    num_epochs = epoch,
                    checkpoint_name = 'complex-biokg_checkpoint.pt',
                    checkpoint_directory = 'kg_checkpoints'
        )

    )
    
    hits_at_10.append(result.get_metric('hits@10'))
    mrr.append(result.get_metric('mrr'))
    
    print('epoch', epoch)
    print('hits@10:', hits_at_10)
    print('mrr:', mrr)
    print('-------------------------------------------------------')

metrices = pd.DataFrame({'epoch': epochs, 'hits@10': hits_at_10, 'mrr': mrr})  
metrices.to_csv('biokg-complex-metrices.csv')

    
dataset = get_dataset(dataset="BioKG")
pack = predict_triples(model=result.model, triples=dataset.validation) 
df = pack.process(factory=result.training).df
df = df.nlargest(n=20, columns="score")
df.to_csv('biokg-complex-scores.csv')


result_df = result.metric_results.to_df()
result_df.to_csv('biokg-complex-results.csv')
