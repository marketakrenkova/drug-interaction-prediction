#!/usr/bin/python3

import pandas as pd
import sys

from pykeen.models import predict
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.models import TransE, NodePiece
from pykeen.evaluation import RankBasedEvaluator


def convert_to_triples_factory(data):
    tf_data = TriplesFactory.from_labeled_triples(
        data[["head", "relation", "tail"]].values,
        create_inverse_triples=True,
        entity_to_id=None,
        relation_to_id=None,
        compact_id=False 
    )

    return tf_data


# def convert_to_triples_factory(data, num_entities, num_relations):   
#     tf_data = CoreTriplesFactory(
#         data,
#         num_entities = num_entities,
#         num_relations = num_relations
#     )

#     return tf_data
      

class DataLoader():
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load(self):
        train_df = pd.read_csv(self.data_dir + 'train.tsv', sep='\t', index_col=[0], engine='python')    
        valid_df = pd.read_csv(self.data_dir + 'valid.tsv', sep='\t', index_col=[0], engine='python')  
        test_df = pd.read_csv(self.data_dir + 'test.tsv', sep='\t', index_col=[0], engine='python')  
   
        # TODO: deal with int values
        self.train = convert_to_triples_factory(train_df.astype(str))
        self.valid = convert_to_triples_factory(valid_df.astype(str))
        self.test = convert_to_triples_factory(test_df.astype(str))


class KG_model:
    def __init__(self, model_name, train_data, valid_data, test_data, specification):
        self.model_name = model_name
        self.train_tf = train_data
        self.valid_tf = valid_data
        self.test_tf = test_data
        self.specification = specification

    def set_params(self, num_epochs, optimizer, evaluator, device):
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.device = device    

    def __str__(self):  
        print(f'Training {self.model_name} - {self.specification} for {self.num_epochs} epochs on {self.device}.')

    def train(self):
        self.trained_model = pipeline(
            training = self.train_tf,
            testing = self.test_tf,
            validation = self.valid_tf,
            model = self.model_name,
            optimizer = self.optimizer,
            evaluator = self.evaluator,
            epochs = self.num_epochs,
            device = self.device,
            training_kwargs = dict(
                num_epochs = self.num_epochs,
                checkpoint_name = self.model_name + '-' + self.specification + '_checkpoint.pt',
                checkpoint_directory = 'kg_checkpoints'
            )
        )  

    def predict_head(self, tail, relation):
        prediction_dir = '../predictions/'
        
        # TODO: use predict method of the model - return a score for a given batch of triplets (or its variants)

        predicted_tails_df = predict.get_head_prediction_df(
            model = self.trained_model.model, 
            tail_label = tail, 
            relation_label = relation, 
            triples_factory = self.trained_model.training,
        )
        print('Leuprolide - decrease_adverse_effects:')
        print(predicted_tails_df.head(10))
        predicted_tails_df = predicted_tails_df.head(50)
        predicted_tails_df.to_csv(prediction_dir + self.model_name + '_' + tail + '_' + relation + '_' + self.specification + '.csv')

# ----------------------------

def main(model_name, specification):

    print('Reading data...')
    data = DataLoader('../data/triplets/')
    data.load()

    # kg = KG_model(model_name, data.train, data.valid, data.test, specification)
    # kg.set_params(5, 'Adam', RankBasedEvaluator, 'gpu')
    # kg.__str__()
    # print()
    # print('Training model...')
    # kg.train()
    # print('Training done.')

    # kg.trained_model.save_to_directory(f'results/results-{model_name}_{specification}')

    # kg.predict_head('Leuprolide', 'decrease_adverse_effects')

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Specify a model name and specification')

    model_name = sys.argv[1]
    specification = sys.argv[2]
    
    main(model_name, specification)