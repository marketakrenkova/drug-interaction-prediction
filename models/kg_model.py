#!/usr/bin/python3

import pandas as pd
import sys
import argparse

import torch

from pykeen.predict import predict_target, predict_triples
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.evaluation import RankBasedEvaluator


def convert_to_triples_factory(data):
    tf_data = TriplesFactory.from_labeled_triples(
        data[["head", "relation", "tail"]].values,
        create_inverse_triples=False,
        entity_to_id=None,
        relation_to_id=None,
        compact_id=False 
    )

    return tf_data

class DataLoader():
    def __init__(self, data_dir, data_name):
        self.data_dir = data_dir
        self.data_name = data_name 

    def load(self):
        train_df = pd.read_csv(self.data_dir + 'train_' + self.data_name + '.tsv', sep='\t', engine='python')    # index_col=[0]
        valid_df = pd.read_csv(self.data_dir + 'valid_' + self.data_name + '.tsv', sep='\t', engine='python')  
        test_df = pd.read_csv(self.data_dir + 'test_' + self.data_name + '.tsv', sep='\t', engine='python')  
   
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
        self.margin = args.loss_margin
        self.negative_sampler = args.neg_sampler
        self.num_neg_per_pos = args.num_neg_per_pos

    # use if I call the model from jupyter notebook, where I don't have args
    def set_params2(self, params):
        self.num_epochs = params['epochs']
        self.optimizer = params['optimizer']
        self.learning_rate = params['learning_rate']
        self.evaluator = RankBasedEvaluator
        self.loss = params['loss']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
        self.batch_size = params['batch']
        self.embedding_dim = params['embedding_dim']
        self.margin = params['margin']

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
            loss_kwargs = dict(
                margin = self.margin
            ),
            optimizer = self.optimizer,
            optimizer_kwargs = dict(
                lr = self.learning_rate
            ),
            negative_sampler = self.negative_sampler,
            negative_sampler_kwargs = dict(
                num_negs_per_pos = self.num_neg_per_pos
            ),
            evaluator = self.evaluator,
            device = self.device,
            training_kwargs = dict(
                batch_size = self.batch_size,
                num_epochs = self.num_epochs,
                checkpoint_name = self.model_name + '-' + self.specification + '_checkpoint.pt',
                checkpoint_frequency=10,
                checkpoint_directory = 'kg_checkpoints'
            ),
            evaluation_kwargs = dict(
                batch_size = 16
            )
        )  

    def predict_tail(self, head, relation, filter_known=False):
        prediction_dir = '../predictions/'

        try:
            pred = predict_target(
                model = self.trained_model.model, 
                head = head, 
                relation = relation, 
                triples_factory = self.trained_model.training,
            )

            if filter_known:
                pred_filtered = pred.filter_triples(self.train_tf)
                pred = pred_filtered.add_membership_columns(validation=self.valid_tf, testing=self.test_tf).df

            predicted_tails_df = pred.head(100)
            predicted_tails_df.to_csv(prediction_dir + self.model_name + '_' + head + '_' + relation + '_' + self.specification + '.csv')
            
        except:
            print(f'No item with id {head} in the dataset.')

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
    
    PREDICT = False

    print('Reading data...')
    data = DataLoader('../data/triplets/', args.data_name)
    # data = DataLoader('../data/dataset-ogb/ogbl_biokg-my_split/')
    data.load()

    kg = KG_model(args.model, data.train, data.valid, data.test, args.model_specification)
    kg.set_params(args)
        
    kg.__str__()
    print()
    print('Training model...')
    kg.train()
    print('Training done.')

    kg.trained_model.save_to_directory(f'results/results-{args.model}_{args.model_specification}')
    
    kg.scores_for_test_triplets(data.test, k=100)
    
    if PREDICT:
        common_drugs = pd.read_csv('../data/common_drugs_num_interactions.csv', sep=';')
        common_drugs = common_drugs.dropna()
        print(common_drugs.head(10))
        common_drugs = common_drugs['db_id'].values
    
<<<<<<< HEAD
        for d in common_drugs:
            kg.predict_tail(d, 'interacts', filter_known=True)
=======
    for d in common_drugs:
        kg.predict_tail(d, 'interacts', filter_known=True)
>>>>>>> af1f1bf05939d59fffa98a301cd66975a330dae0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KG training')
    parser.add_argument('-d', '--data_name', type=str, default='')
    parser.add_argument('-m', '--model', type=str, default='TransE')
    parser.add_argument('-s', '--model_specification', type=str, default='')
    parser.add_argument('-emb', '--embedding_dim', type=int, default=32)
    parser.add_argument('-e', '--epochs', type=int, default=1)
    parser.add_argument('-o', '--optimizer', type=str, default='Adam')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
    parser.add_argument('-l', '--loss', type=str, default='MarginRankingLoss')
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-lm', '--loss_margin', type=float, default=1.0)
    parser.add_argument('-neg', '--neg_sampler', type=str, default='basic')
    parser.add_argument('-nn', '--num_neg_per_pos', type=int, default='1')
    

    args = parser.parse_args()
    print(args)

    main(args)
