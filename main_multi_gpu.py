import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.optim
import torchmetrics.functional as MF
import dgl
import dgl.function as fn
import dgl.nn as dglnn
from dgl.multiprocessing import shared_tensor
import time
import numpy as np
# from ogb.nodeproppred import DglNodePropPredDataset
# import tqdm
th = torch
import gipa_model
import utils

import sys

def rocauc(pred, labels, evaluator):
    return evaluator.eval({"y_pred": pred, "y_true": labels})["rocauc"]

def preprocess(graph, use_label=False):
    # add additional features
    graph.update_all(fn.copy_e("feat", "e"), fn.sum("e", "feat_add"))
    if use_label:
        graph.ndata['feat'] = th.cat((graph.ndata['feat_add'], graph.ndata['feat']), dim=1)
    else:
        graph.ndata['feat'] = graph.ndata['feat_add']
    graph.create_formats_()

    return graph

def train(rank, world_size, graph, num_classes, n_node_feat, n_edge_feat, split_idx, evaluator, args):
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', 'tcp://127.0.0.1:12347', world_size=world_size, rank=rank)
    utils.seed(args.seed)
    model = gipa_model.GIPA(n_node_feat=n_node_feat,
                            n_edge_feat=n_edge_feat,
                            n_node_emb=args.n_node_emb,
                            n_edge_emb=args.n_edge_emb,
                            n_hiddens_att=args.n_hiddens_att,
                            n_heads_att=args.n_heads_att,
                            n_hiddens_prop=args.n_hiddens_prop,
                            n_hiddens_agg=args.n_hiddens_agg,
                            n_hiddens_deep=args.n_hiddens_deep,
                            n_layers=args.n_layers,
                            n_classes=num_classes,
                            agg_type=args.agg_type,
                            act_type=args.act_type,
                            edge_drop=args.edge_drop,
                            dropout_node=args.dropout_node,
                            dropout_att=args.dropout_att,
                            dropout_prop=args.dropout_prop,
                            dropout_agg=args.dropout_agg,
                            dropout_deep=args.dropout_deep).cuda()
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.75, patience=50,
                                                           verbose=True)
    train_idx, valid_idx, test_idx = split_idx

    # move ids to GPU
    train_idx = train_idx.to('cuda')
    valid_idx = valid_idx.to('cuda')
    test_idx = test_idx.to('cuda')

    # For training, each process/GPU will get a subset of the
    # train_idx/valid_idx, and generate mini-batches indepednetly. This allows
    # the only communication neccessary in training to be the all-reduce for
    # the gradients performed by the DDP wrapper (created above).
    sampler = dgl.dataloading.NeighborSampler(
            [args.sampling for _ in range(args.n_layers)], prefetch_node_feats=['feat'], prefetch_labels=['label'])
    train_dataloader = dgl.dataloading.DataLoader(
            graph, train_idx, sampler,
            device='cuda', batch_size=args.batch_size, shuffle=True, drop_last=False,
            num_workers=0, use_ddp=True, use_uva=True)
    valid_dataloader = dgl.dataloading.DataLoader(
            graph, valid_idx, sampler, device='cuda', batch_size=1024, shuffle=True,
            drop_last=False, num_workers=0, use_ddp=True,
            use_uva=True)
    test_dataloader = dgl.dataloading.DataLoader(
            graph, test_idx, sampler, device='cuda', batch_size=1024, shuffle=True,
            drop_last=False, num_workers=0, use_ddp=True,
            use_uva=True)
    loss_fcn = nn.BCEWithLogitsLoss()
    durations = []
    best_val_score = 0.0
    final_test_score = 0.0
    for epoch in range(args.n_epochs):
        dist.barrier()
        model.train()
        t0 = time.time()
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            subgraphs = blocks
            batch_ids = th.arange(len(output_nodes))
#             print(model.device, subgraphs[0].device, batch_ids.device)
            pred = model(subgraphs)[batch_ids]
            label = subgraphs[-1].dstdata['label'][batch_ids].float()
#             print(pred.shape, label.shape)
            batch_loss = loss_fcn(pred, label)
            opt.zero_grad()
            batch_loss.backward()
            opt.step()
#             if it % 20 == 0 and rank == 0:
#                 acc = MF.accuracy(y_hat, y)
#                 mem = torch.cuda.max_memory_allocated() / 1000000
#                 print('Loss', loss.item(), 'Acc', acc.item(), 'GPU Mem', mem, 'MB')
        tt = time.time()

        if rank == 0:
            print(f'epoch: {epoch}, time: {tt - t0}, loss: {batch_loss.cpu().item()}, lr: {opt.state_dict()["param_groups"][0]["lr"]}')
        durations.append(tt - t0)
        if epoch % args.eval_every == 0:
            model.eval()
            @torch.no_grad()
            def eval(dataloader):
                ys = []
                y_hats = []
                for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                    with torch.no_grad():
                        # x = blocks[0].srcdata['feat']
                        ys.append(blocks[-1].dstdata['label'])
                        y_hats.append(model.module(blocks))
                score = evaluator.eval({"y_pred": torch.cat(y_hats), "y_true": torch.cat(ys)})["rocauc"]

                score = torch.tensor(score / world_size).cuda()
    #             print(val_score)
                dist.all_reduce(score)
                dist.barrier()
                return score.item()
            val_score, test_score = eval(valid_dataloader), eval(test_dataloader)
            lr_scheduler.step(val_score)
            
            if rank == 0:
                print(f'Validation acc: {val_score}, test acc: {test_score}')
                if val_score > best_val_score:
                    if test_score > 0.85:
#                         print('full eval')
#                         test_score = evaluate_test(model, graph, graph.ndata['label'], test_idx, evaluator)
                        th.save(model.state_dict(),
                               f'./saved_models/gipa_proteins_ep{epoch}.pt')

    # if rank == 0:
    #     print(np.mean(durations[4:]), np.std(durations[4:]))
    # model.eval()
    # with torch.no_grad():
    #     # since we do 1-layer at a time, use a very large batch size
    #     pred = model.module.inference(graph, device='cuda', batch_size=2**16)
    #     if rank == 0:
    #         acc = MF.accuracy(pred[test_idx], graph.ndata['label'][test_idx])
    #         print('Test acc:', acc.item())
# print(1)
print(__name__)
if __name__ == '__main__':
#     print(2)
#     sys.argv = 'python --n-layers 6 --n-epochs 8000 --lr 0.01 --batch-size 12000 --use-label --if-save --preprocess --n-hop 1 --gpu 0 --eval-every 5 --seed 0'.split(' ')

    parser = argparse.ArgumentParser(description='GIPA')
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--dataset", type=str, default='proteins',
                        help="dataset: cora, citeseer, pubmed, amazon, reddit, proteins")
    parser.add_argument("--data-root", type=str, default='/home/ogb/data/ogb/proteins/',
                        help="dataset download dir")
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--n-layers", type=int, default=6, help="number of gipa layers")
    parser.add_argument("--n-epochs", type=int, default=7000, help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1000, help="mini-batch size")
    parser.add_argument("--sampling", type=int, default=16, help="sampling size")
    parser.add_argument("--n-hop", type=int, default=1, help="number of hops")
    parser.add_argument("--weight-decay", type=float, default=0, help="Weight for L2 loss")
    parser.add_argument("--n-node-emb", type=int, default=80, help="size of node feature embedding")
    parser.add_argument("--dropout-node", type=float, default=0.1, help="dropout probability of node features")
    parser.add_argument("--n-edge-emb", type=int, default=16, help="size of edge feature embedding")
    parser.add_argument("--edge-drop", type=float, default=0.1, help="dropout probability of edge features")

    parser.add_argument("--n-hiddens-att", type=list, default=[80],
                        help="list of number of attention hidden units")
    parser.add_argument("--n-heads-att", type=int, default=8, help="number of attention heads")
    parser.add_argument("--dropout-att", type=float, default=0.1, help="dropout probability of attention layers")
    parser.add_argument("--n-hiddens-prop", type=list, default=[80],
                        help="list of number of propagation hidden units")
    parser.add_argument("--dropout-prop", type=float, default=0.25, help="dropout probability of propagation layers")
    parser.add_argument("--n-hiddens-agg", type=list, default=[],
                        help="list of number of aggregation hidden units")
    parser.add_argument("--dropout-agg", type=float, default=0.25, help="dropout probability of aggregation layers")
    parser.add_argument("--n-hiddens-deep", type=list, default=[], help="list of number of deep hidden units")
    parser.add_argument("--dropout-deep", type=float, default=0.5, help="dropout probability of deep layers")
    parser.add_argument("--eval-every", type=int, default=5, help="evaluation frequency")
    parser.add_argument("--eval-best-every", type=int, default=20, help="best evaluation frequency")

    parser.add_argument("--agg-type", type=str, default='sum', help="aggregation type")
    parser.add_argument("--act-type", type=str, default='relu', help="activation type")
    parser.add_argument("--self-loop", action='store_true', help="graph self-loop (default=False)")
    parser.add_argument("--normalize", action='store_true', help="graph normalization (default=False)")
    parser.add_argument("--preprocess", action='store_true', help="graph preprocessing (default=False)")
    parser.add_argument("--use-label", action='store_true', help="use label as node features (default=False)")
    parser.add_argument("--if-save", action='store_true', help="save the best model (default=False)")

    args = parser.parse_args()
    
    graph, labels, train_idx, val_idx, test_idx, evaluator = utils.load_dataset(args.dataset, root=args.data_root)
    if args.preprocess:
        graph = preprocess(graph, args.use_label)

    if args.self_loop:
        graph = dgl.add_self_loop(graph)

    if args.normalize:
        degs = graph.in_degrees().float()
        norm = th.pow(degs, -0.5)
        norm[th.isinf(norm)] = 0
        norm = norm.to(dev)
        graph.ndata['norm'] = norm.unsqueeze(1)

    graph.create_formats_()     # must be called before mp.spawn().
#     print(len(train_idx))
    # split_idx = dataset.get_idx_split()
    num_classes = labels.shape[1]
    n_node_feat = graph.ndata["feat"].shape[-1]
    n_edge_feat = graph.edata["feat"].shape[-1]
    # use all available GPUs
    n_procs = torch.cuda.device_count()
    # Tested with mp.spawn and fork.  Both worked and got 4s per epoch with 4 GPUs
    # and 3.86s per epoch with 8 GPUs on p2.8x, compared to 5.2s from official examples.
    import torch.multiprocessing as mp
    mp.spawn(train, args=(n_procs, graph, num_classes, n_node_feat, n_edge_feat, (train_idx, val_idx, test_idx), evaluator, args), nprocs=n_procs)