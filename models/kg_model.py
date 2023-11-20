#!/usr/bin/python3

import pandas as pd
import sys
import argparse
import os
#import wandb

import torch

from pykeen.predict import predict_target, predict_triples
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.evaluation import RankBasedEvaluator, OGBEvaluator

from pykeen.models import TransE, ComplEx
from torch.optim import Adam
from pykeen.training import SLCWATrainingLoop


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
        self.data_name = args.data_name

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
        self.negative_sampler = 'basic'
        self.num_neg_per_pos = 30
        self.data_name = params['data_name']

    def __str__(self):  
        print(f'Training {self.model_name} - {self.specification}-{self.data_name} for {self.num_epochs} epochs on {self.device}.')

    def train(self, model_checkpoint_path):
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
            evaluator = None,
            device = self.device,
            training_kwargs = dict(
                batch_size = self.batch_size,
                num_epochs = self.num_epochs,
                checkpoint_name = model_checkpoint_path,
                checkpoint_frequency=10,
                checkpoint_directory = 'kg_checkpoints'
            ),
            evaluation_kwargs = dict(
                batch_size = 16
            ),
            stopper='early',
            stopper_kwargs=dict(frequency=5, patience=2, relative_delta=0.002),
            #result_tracker='wandb',
            #result_tracker_kwargs=dict(
            #    project='kg_drug_interactions',
            #),  
        ) 

    def predict_tail(self, trained_model, triples, head, relation, filter_known=False):
        prediction_dir = '../predictions/' + self.specification + '/' + self.data_name + '/'

        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir)

        try:
            pred = predict_target(
                model = trained_model,
                head = head, 
                relation = relation, 
                triples_factory = triples,
            )

            if filter_known:
                # remove tragets known from training set
                pred_filtered = pred.filter_triples(self.train_tf)
                # add information whether the target is in validation and testing datsets 
                pred = pred_filtered.add_membership_columns(validation=self.valid_tf, testing=self.test_tf)

            predicted_tails_df = pred.df.head(100)
            predicted_tails_df.to_csv(prediction_dir + self.model_name + '_' + head + '_' + relation + '_' + self.specification + '.csv')
        
        except:
            print(f'No item with id {head} in the dataset.')

    # returns scores for given triplets (validation/test) - only k highest scores
    def scores_for_test_triplets(self, test_triplets, k=100):
        prediction_dir = '../predictions/'
        
        pack = predict_triples(model=self.trained_model.model, triples=test_triplets) #self.valid_tf
        df = pack.process(factory=self.trained_model.training).df
        df = df.nlargest(n=k, columns="score")
        df.to_csv(prediction_dir + self.model_name + "_" + self.data_name + '_testset_scores_' + self.specification + '.csv')
        print(df.head())

    def save_metrices(self):
        prediction_dir = '../predictions/'
        metrices = dict()
        metrices['hits@10'] = [self.trained_model.get_metric('hits@10')]
        metrices['mrr'] = [self.trained_model.get_metric('mrr')]

        df_result = pd.DataFrame(metrices)
        print(df_result)
        df_result.to_csv(prediction_dir + self.specification + self.data_name + '_metrices.csv')


def load_model(model_checkpoint_path, model_result_path):
    model = None

    if os.path.exists(model_checkpoint_path) and os.path.exists(model_result_path):
        model = torch.load(model_result_path)
        checkpoint = torch.load(model_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    else:
        print("No trained model at path: {}".format(model_checkpoint_path))

    return model


def main(args):

    PREDICT = True

    print('Reading data...')
    data = DataLoader('../data/triplets/' + args.run + '/', args.data_name)
    data.load()

    model_specification = args.model_specification + '-' + args.run

    kg = KG_model(args.model, data.train, data.valid, data.test, model_specification)
    kg.set_params(args)

    model_checkpoint_path = f'{args.model}-{model_specification}-{args.data_name}_checkpoint.pt'
    model_result_dir = f'results/results-{args.model}_{model_specification}-{args.data_name}'

    # if the model should be trained 
    if not args.not_train:
        kg.__str__()
        print()
        print('Training model...')
        kg.train(model_checkpoint_path)
        print('Training done.')
        # kg.scores_for_test_triplets(data.test, k=100)
        kg.trained_model.save_to_directory(model_result_dir)
        kg.save_metrices()

    if PREDICT:
        loaded_model = None
        if args.not_train:
            print('Loading a trained model...')
            loaded_model = load_model('kg_checkpoints/' + model_checkpoint_path, model_result_dir + '/trained_model.pkl')

        print("Predicting new interactions...")
        common_drugs = pd.read_csv('../data/common_drugs_num_interactions.csv', sep=';')
        #common_drugs = pd.read_csv('../data/drugs4prediction.csv', sep=';')
        common_drugs = common_drugs.dropna()
        common_drugs = common_drugs['db_id'].values

        with open('../data/foods4predictions-2.txt', 'r') as f:
            foods = f.readlines()

        foods = [food.strip() for food in foods]
    
        # drug predictions
        for d in common_drugs[:100]:
            if loaded_model is None:
                kg.predict_tail(kg.trained_model.model, kg.trained_model.training, d, 'interacts', filter_known=True)
            else:
                kg.predict_tail(loaded_model, data.train, d, 'interacts', filter_known=True)

        # food predicitons
        for food in foods: 
            if loaded_model is None:
                kg.predict_tail(kg.trained_model.model, kg.trained_model.training, food, 'interacts', filter_known=True)
            else:
                kg.predict_tail(loaded_model, data.train, food, 'interacts', filter_known=True)


        print('DONE')


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
    parser.add_argument("--not_train", action="store_true")
    parser.add_argument("-r", "--run", type=str, default='run1')
    

    args = parser.parse_args()
    print(args)

    #os.environ["WANDB_API_KEY"] = "a0dcca4cf18920b5c23ec09023f46ffa76caad5b"
    #wandb.login()

    main(args)

