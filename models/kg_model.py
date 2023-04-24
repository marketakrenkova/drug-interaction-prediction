#!/usr/bin/python3

import pandas as pd
import sys
import argparse

import torch

from pykeen.predict import predict_target, predict_triples, predict_all
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

class DataLoader():
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load(self):
        train_df = pd.read_csv(self.data_dir + 'train.tsv', sep='\t', engine='python')    # index_col=[0]
        valid_df = pd.read_csv(self.data_dir + 'valid.tsv', sep='\t', engine='python')  
        test_df = pd.read_csv(self.data_dir + 'test.tsv', sep='\t', engine='python')  
   
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

    def set_params(self, args):
        self.num_epochs = args.epochs
        self.optimizer = args.optimizer
        self.learning_rate = args.learning_rate
        self.evaluator = RankBasedEvaluator
        self.loss = args.loss 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
        self.batch_size = args.batch_size
        self.embedding_dim = args.embedding_dim

    def __str__(self):  
        print(f'Training {self.model_name} - {self.specification} for {self.num_epochs} epochs on {self.device}.')

    def train(self):
        self.trained_model = pipeline(
            training = self.train_tf,
            testing = self.test_tf,
            validation = self.valid_tf,
            model = self.model_name,
            model_kwargs = dict(
                embedding_dim = self.embedding_dim
            ),
            loss = self.loss,
            optimizer = self.optimizer,
            optimizer_kwargs = dict(
                lr = self.learning_rate
            ),
            evaluator = self.evaluator,
            device = self.device,
            training_kwargs = dict(
                batch_size = self.batch_size,
                num_epochs = self.num_epochs,
                checkpoint_name = self.model_name + '-' + self.specification + '_checkpoint.pt',
                checkpoint_directory = 'kg_checkpoints'
            )
        )  

    def predict_tail(self, head, relation, filter_known=False):
        prediction_dir = '../predictions/'

        pred = predict_target(
            model = self.trained_model.model, 
            head = head, 
            relation = relation, 
            triples_factory = self.trained_model.training,
        )
        
        if filter_known:
            pred_filtered = pred.filter_triples(self.train_tf)
            pred = pred.add_membership_columns(validation=self.valid_tf, testing=self.test_tf).df
        
        print('Leuprolide - decrease_adverse_effects:')
        print(pred.head(10))
        predicted_tails_df = pred.head(50)
        predicted_tails_df.to_csv(prediction_dir + self.model_name + '_' + head + '_' + relation + '_' + self.specification + '.csv')

    # returns scores for given triplets (validation/test) - only k highest scores
    def scores_for_test_triplets(self, test_triplets, k=100):
        prediction_dir = '../predictions/'
        
        pack = predict_triples(model=self.trained_model.model, triples=test_triplets) #self.valid_tf
        df = pack.process(factory=self.trained_model.training).df
        df = df.nlargest(n=k, columns="score")
        df.to_csv(prediction_dir + self.model_name + '_testset_scores_' + self.specification + '.csv')
        print(df.head())


#     # predicts all posible new triplets, stores just k triplets with highest scores
#     # computationally expensive!!!    
#     def predict_all_triplets(self, k=100):
#         pack = predict_all(model=self.trained_model.model, k)
#         pred = pack.process(factory=result.training)
#         pred_annotated = pred.add_membership_columns(training=result.training)
#         return pred_annotated.df
    
# ----------------------------

def main(args):

    print('Reading data...')
    data = DataLoader('../data/triplets/')
    data.load()

    kg = KG_model(args.model, data.train, data.valid, data.test, args.model_specification)
    kg.set_params(args)
        
    kg.__str__()
    print()
    print('Training model...')
    kg.train()
    print('Training done.')

    kg.trained_model.save_to_directory(f'results/results-{args.model}_{args.model_specification}')
    
    kg.predict_tail('DB00007', 'decrease_adverse_effects', filter_known=True)
    kg.scores_for_test_triplets(data.test, k=100)

if __name__ == '__main__':
#     if len(sys.argv) < 3:
#         print('Specify a model name and specification')

#     model_name = sys.argv[1]
#     specification = sys.argv[2]
    
    parser = argparse.ArgumentParser(description='KG training')
    parser.add_argument('-m', '--model', type=str, default='TransE')
    parser.add_argument('-s', '--model_specification', type=str, default='')
    parser.add_argument('-emb', '--embedding_dim', type=int, default=32)
    parser.add_argument('-e', '--epochs', type=int, default=1)
    parser.add_argument('-o', '--optimizer', type=str, default='Adam')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
    parser.add_argument('-l', '--loss', type=str, default='MarginRankingLoss')
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    
    
    
    # TODO: more args??

    args = parser.parse_args()
    print(args)

    main(args)
