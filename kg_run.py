#!/usr/bin/python3

import pandas as pd

from pykeen.models import predict
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.models import TransE
from pykeen.evaluation import RankBasedEvaluator


def convert_to_triples_factory(data):
    tf_data = TriplesFactory.from_labeled_triples(
      data[["head", "relation", "tail"]].values,
      create_inverse_triples=False,
      entity_to_id=None,
      relation_to_id=None,
      compact_id=False 
    )
    #print(tf_data)  # kam mizeji nejake trojice? - jiny pocet zde a po vytvoreni datasetu
    return tf_data

def train_model(tf_train, tf_valid, tf_test):
    # creating a model
    result = pipeline(
        training=tf_train,
        testing=tf_test,
        validation=tf_valid,
        model='TransE',
        epochs=2,
        evaluator=RankBasedEvaluator,
        device='gpu'
    )
    return result

def main():
    print('Reading data...')
    train = pd.read_csv('data/triplets/train.tsv', sep='\t', index_col=[0], engine='python')
    valid = pd.read_csv('data/triplets/valid.tsv', sep='\t', index_col=[0], engine='python')
    test = pd.read_csv('data/triplets/test.tsv', sep='\t', index_col=[0], engine='python')

    print('Transforming...')
    tf_train = convert_to_triples_factory(train.astype(str))
    tf_valid = convert_to_triples_factory(valid.astype(str))
    tf_test = convert_to_triples_factory(test.astype(str))
    
    tf_train, _ = tf_train.split(0.15)
    tf_valid, _ = tf_valid.split(0.15)
    tf_test, _ = tf_test.split(0.15)

    print('Training model...')
    results = train_model(tf_train, tf_valid, tf_test)
    print('Training done.')
    
    hits_at_10 = results.get_metric('hits@10')
    print('Hits at 10:', hits_at_10)
 
    results.save_to_directory("results")

    with open('out.txt', 'w') as f:
        f.write(hits_at_10)

if __name__ == "__main__":
    main()    
    
