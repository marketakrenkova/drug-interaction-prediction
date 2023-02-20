#!/usr/bin/python3

import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling
import torch_geometric.transforms as T

from ogb.linkproppred import Evaluator, PygLinkPropPredDataset

from pykeen.evaluation import RankBasedEvaluator
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline, pipeline_from_config 


def convert_to_triples_factory(data):
    tf_data = TriplesFactory.from_labeled_triples(
        data[["head", "relation", "tail"]].values,
        create_inverse_triples=True,
        entity_to_id=None,
        relation_to_id=None,
        compact_id=False 
    )

    print(tf_data.mapped_triples)

    return tf_data

def load_data():
    dataset = PygLinkPropPredDataset(name='ogbl-ddi', transform=T.ToSparseTensor())
    
    split_edge = dataset.get_edge_split()
    train_edge, valid_edge, test_edge = split_edge["train"], split_edge["valid"], split_edge["test"]
    
    # add relation type - interacts with
    train = train_edge['edge']
    train = torch.tensor([[x[0], 0, x[1]] for x in train])
#     train_df = pd.DataFrame(train, columns=['head', 'relation', 'tail']).astype(str)

    valid = valid_edge['edge']
    valid = torch.tensor([[x[0], 0, x[1]] for x in valid])
#     valid_df = pd.DataFrame(valid, columns=['head', 'relation', 'tail']).astype(str)

    valid_neg = valid_edge['edge_neg']
    valid_neg = torch.tensor([[x[0], 0, x[1]] for x in valid_neg])

    test = test_edge['edge']
    test = torch.tensor([[x[0], 0, x[1]] for x in test])
#     test_df = pd.DataFrame(test, columns=['head', 'relation', 'tail']).astype(str)

    test_neg = test_edge['edge_neg']
    test_neg = torch.tensor([[x[0], 0, x[1]] for x in test_neg])

#     train_tf = convert_to_triples_factory(train_df)
#     valid_tf = convert_to_triples_factory(valid_df)
#     test_tf = convert_to_triples_factory(test_df)
    
    return train, valid, valid_neg, test, test_neg


def save_to_txt(dir_data_my_split, train_df, valid_df, test_df):

    train_df.to_csv(dir_data_my_split + 'train.txt', sep='\t', header=False, index=False)
    valid_df.to_csv(dir_data_my_split + 'valid.txt', sep='\t', header=False, index=False)
    test_df.to_csv(dir_data_my_split + 'test.txt', sep='\t', header=False, index=False)
    
def compute_scores(trained_model, train, valid, valid_neg, test, test_neg):
    # compute scores for positive and negative triplets 
    batch_size = 512

    n = train.size(0) // batch_size
    pos_train_preds = []
    for i in range(n+1):
        start_idx = i*batch_size
        end_idx = min((i+1)*batch_size, train.size(0))
        edge = train[start_idx:end_idx]
        pos_train_preds += [trained_model.model.score_hrt(edge).squeeze().cpu().detach()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    n = valid.size(0) // batch_size
    pos_valid_preds = []
    for i in range(n+1):
        start_idx = i*batch_size
        end_idx = min((i+1)*batch_size, valid.size(0))
        edge = valid[start_idx:end_idx]
        pos_valid_preds += [trained_model.model.score_hrt(edge).squeeze().cpu().detach()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    n = valid_neg.size(0) // batch_size
    neg_valid_preds = []
    for i in range(n+1):
        start_idx = i*batch_size
        end_idx = min((i+1)*batch_size, valid_neg.size(0))
        edge = valid_neg[start_idx:end_idx]
        neg_valid_preds += [trained_model.model.score_hrt(edge).squeeze().cpu().detach()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    n = test.size(0) // batch_size
    pos_test_preds = []
    for i in range(n+1):
        start_idx = i*batch_size
        end_idx = min((i+1)*batch_size, test.size(0))
        edge = test[start_idx:end_idx]
        pos_test_preds += [trained_model.model.score_hrt(edge).squeeze().cpu().detach()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    n = test_neg.size(0) // batch_size
    neg_test_preds = []
    for i in range(n+1):
        start_idx = i*batch_size
        end_idx = min((i+1)*batch_size, test_neg.size(0))
        edge = test_neg[start_idx:end_idx]
        neg_test_preds += [trained_model.model.score_hrt(edge).squeeze().cpu().detach()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)
    
    return pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred

def evaluate_ogb(pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred):
    evaluator = Evaluator(name = 'ogbl-ddi')

    results = {}
    for K in [10, 20, 30]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    for hits, result in results.items():
        print(hits)
        train_hits, valid_hits, test_hits = result
        print(f'Train: {100 * train_hits:.2f}%')
        print(f'Valid: {100 * valid_hits:.2f}%')
        print(f'Test: {100 * test_hits:.2f}%')
        print()



def main():
    print('Loading data...')
    train, valid, valid_neg, test, test_neg = load_data()
    
    train_df = pd.DataFrame(train, columns=['head', 'relation', 'tail']).astype(str)
    valid_df = pd.DataFrame(valid, columns=['head', 'relation', 'tail']).astype(str)
    test_df = pd.DataFrame(test, columns=['head', 'relation', 'tail']).astype(str)
    
    dir_data_my_split = 'dataset/ogbl_ddi-my_split/'
    save_to_txt(dir_data_my_split, train_df, valid_df, test_df)
    
    config = {
        'metadata': dict(
            title='HolE'
        ),
        'pipeline': dict(
            training = 'dataset/ogbl_ddi-my_split/train.txt',
            validation = 'dataset/ogbl_ddi-my_split/valid.txt',
            testing = 'dataset/ogbl_ddi-my_split/test.txt',
            model='HolE',
            model_kwargs=dict(
                   embedding_dim=96,
            ),
            optimizer='SGD',
            optimizer_kwargs=dict(lr=0.09),
            loss='marginranking',
            loss_kwargs=dict(margin=2.64),
            training_loop='slcwa',
            training_kwargs=dict(
                num_epochs=20, 
                batch_size=64, 
                checkpoint_name='HolE_checkpoint-ogb-20.pt',
                checkpoint_directory='kg_checkpoints',
                checkpoint_frequency=5    
            ),
            negative_sampler='basic',
            negative_sampler_kwargs=dict(num_negs_per_pos=94),
            evaluator='rankbased',
            evaluator_kwargs=dict(filtered=True),
            evaluation_kwargs=dict(batch_size=64),
            stopper='early',
            stopper_kwargs=dict(
                patience=10,
                relative_delta=0.002
            ),
            result_tracker='tensorboard',
            result_tracker_kwargs=dict(experiment_name='hole-ogb-20', experiment_path='pykeen_logs')
        )
    }
    
    print('Training...')
    pipeline_result = pipeline_from_config(config)
    print('Training done.')

    print('Evaluation...')
    pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred = compute_scores(pipeline_result, train, valid, valid_neg, test, test_neg)
    
    evaluate_ogb(pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    


if __name__ == "__main__":
    main()


