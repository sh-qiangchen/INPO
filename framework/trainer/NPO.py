import os
import time
import dgl.data
import wandb
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import GraphSAINTRandomWalkSampler

from .base import Trainer, KGTrainer
from ..evaluation import *
from ..utils import *
from ..nora import nora
from dataclasses import dataclass

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

@dataclass
class Args:
    num_layers: int = 2
    self_buff: int = 8
    k1: float = 1.0
    k2: list = (0.5, 0.0) 
    k3: int = 1

def weight(model):
    t = 0
    for p in model.parameters():
        t += torch.norm(p)
    
    return t

class NPOTrainer(Trainer):

    def train(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        if 'ogbl' in self.args.dataset:
            args.eval_on_cpu = False
            return self.train_fullbatch(model, data, optimizer, args, logits_ori, attack_model_all, attack_model_sub)

        else:
            return self.train_fullbatch(model, data, optimizer, args, logits_ori, attack_model_all, attack_model_sub)

    def train_fullbatch(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        model = model.to(device)
        data = data.to(device)
        
        start_time = time.time()
        best_metric = 0

        # MI Attack before unlearning
        if attack_model_all is not None:
            mi_logit_all_before, mi_sucrate_all_before = member_infer_attack(model, attack_model_all, data)
            self.trainer_log['mi_logit_all_before'] = mi_logit_all_before
            self.trainer_log['mi_sucrate_all_before'] = mi_sucrate_all_before
        if attack_model_sub is not None:
            mi_logit_sub_before, mi_sucrate_sub_before = member_infer_attack(model, attack_model_sub, data)
            self.trainer_log['mi_logit_sub_before'] = mi_logit_sub_before
            self.trainer_log['mi_sucrate_sub_before'] = mi_sucrate_sub_before
        
        # original_path = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, 'original', str(args.random_seed))
        # if os.path.exists(os.path.join(original_path, 'model_best.pt')):
        #     model_ckpt_sft = torch.load(os.path.join(original_path, 'model_best.pt'), map_location=device)
        #     model.load_state_dict(model_ckpt_sft['model_state'], strict=False)
            
        # Original node embeddings
        with torch.no_grad():
            node_embedding_origin = model(data.x, data.train_pos_edge_index, return_all_emb=False)  
            ref_logits = model.decode(node_embedding_origin, data.train_pos_edge_index).sigmoid()
            ref_w = ref_logits[data.dr_mask][:sum(data.df_mask)] 
            ref_l = ref_logits[data.df_mask] 
        
        # Fast Inference of Removal-Based Node Influence
        infer_hp = Args()     
        import dgl
        num_nodes = data.x.size(0)
        src, dst = data.edge_index
        graph_dgl = dgl.graph((src, dst), num_nodes=num_nodes)
        graph_dgl.ndata['feat'] = data.x.clone()  # deepcopy
        influence_score = torch.from_numpy(nora(infer_hp, model, graph_dgl, data, device)).to(device)   


        for epoch in trange(args.epochs, desc='Unlearning'):
            model.train()

            start_time = time.time()

            # Positive and negative sample
            neg_edge_index = negative_sampling(
                edge_index=data.train_pos_edge_index[:, data.df_mask],
                num_nodes=data.num_nodes,
                num_neg_samples=data.df_mask.sum())

            # z = model(data.x, data.train_pos_edge_index)
            # NPO modification
            z = model(data.x, data.train_pos_edge_index, df_mask=data.train_pos_edge_index[:, data.dr_mask], edge_influence=influence_score)
            logits = model.decode(z, data.train_pos_edge_index[:, data.df_mask]).sigmoid()
            # label = torch.ones_like(logits, dtype=torch.float, device='cuda')
            # loss = -F.binary_cross_entropy_with_logits(logits, label)         
            
            # Grad Descent loss on retain set
            rt_logit = model.decode(z, data.train_pos_edge_index[:, data.dr_mask]).sigmoid()
            rt_label = torch.ones_like(rt_logit, dtype=torch.float, device='cuda')
            GD_loss = F.binary_cross_entropy_with_logits(rt_logit, rt_label)
                
                
            edge_influence = (influence_score[data.train_pos_edge_index[:, data.df_mask][0]] \
                + influence_score[data.train_pos_edge_index[:, data.df_mask][1]]).to(device)

            # NPO
            beta = 5 # DBLP: beta = 0.5
            S = 2 * torch.pow(logits, beta) / (torch.pow(logits, beta) + torch.pow(ref_l, beta)) 
            if epoch in [1, 2, 200, 250]:
                import matplotlib.pyplot as plt
                # S_numpy = S.cpu().detach().numpy()
                plt.figure(figsize=(10, 6))
                plt.plot(S.cpu().detach().numpy(), marker='o', linestyle='', alpha=0.7)
                plt.tick_params(axis='y', which='major', labelsize=30) 
                plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) 
                # plt.legend(loc='lower right', fontsize=28)
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(str(epoch) + '_S_dist_MPNN.pdf')
            
           
            loss_NPO = 2 * torch.mean(torch.log(1 + torch.pow(logits / ref_l + 0.0001, beta))) / beta # avoid nan when logits / ref_l = 0
            # loss_NPO = -2 * torch.mean(torch.log((-beta * torch.log(logits + 0.0001)).sigmoid())) / beta 


            # # check distribution
            # import matplotlib.pyplot as plt
            # import random
            # num_samples = sum(data.df_mask)
            # if epoch in [0, 10, 100, 200, 250, 300]:
            #     plt.figure(figsize=(10, 6))
            #     sampled_rt_logit = random.sample(list(rt_logit.detach().cpu().numpy()), num_samples)
            #     plt.hist(sampled_rt_logit, bins=50, alpha=0.7, color='blue', label='rt_logit')
            #     plt.hist(logits.detach().cpu().numpy(), bins=50, alpha=0.7, color='green', label='logits')
            #     plt.title('Distribution of rt_logit and logits')
            #     plt.xlabel('Value')
            #     plt.ylabel('Frequency')
            #     plt.legend()
            #     plt.tight_layout()
            #     # plt.show()
            #     plt.savefig(str(epoch) + '_contrast_dist.pdf')
            
        
            # Local-structure Alignment
            structure_embedd_ori = torch.cat([node_embedding_origin[data.train_pos_edge_index[:, data.df_mask][0]], node_embedding_origin[data.train_pos_edge_index[:, data.df_mask][1]]], dim=0)
            structure_embedd = torch.cat([z[data.train_pos_edge_index[:, data.df_mask][0]], z[data.train_pos_edge_index[:, data.df_mask][1]]], dim=0)
            TE = torch.mean(-torch.sum(F.softmax(structure_embedd_ori, dim=1) * torch.log(F.softmax(structure_embedd, dim=1)), dim=1))
        
            # balance loss
            # alpha = 1.6
            loss = loss_NPO + GD_loss + 0.5 * TE
            # loss = loss_NPO + GD_loss
            # loss = loss_NPO + 1.6 * GD_loss - 3 * TE   # DBLP
            print("\nGD_loss: " + str(GD_loss) + "======NPO_loss: " + str(loss_NPO) + "======TE_loss: " + str(TE))
            
            # KTO
            # aligned and unaligned samples
            # rt_aligned_mask = rt_logit > 0.6
            # policy_chosen_logps = rt_logit[rt_aligned_mask]
            # reference_chosen_logps = ref_logits[data.dr_mask][rt_aligned_mask]
            # policy_rejected_logps = rt_logit[~rt_aligned_mask]
            # reference_rejected_logps = ref_logits[data.dr_mask][~rt_aligned_mask]
            
            # ft_aligned_mask = logits < 0.5
            # policy_chosen_logps = logits[ft_aligned_mask]
            # reference_chosen_logps = ref_l[ft_aligned_mask]
            # policy_rejected_logps = logits[~ft_aligned_mask]
            # reference_rejected_logps = ref_l[~ft_aligned_mask]
            
            
            # # KTO reference constant z0
            # KL_rewards = torch.cat((policy_chosen_logps, policy_rejected_logps), 0).sum(-1) - torch.cat((reference_chosen_logps, reference_rejected_logps), 0).sum(-1)
            # z0 = KL_rewards.clamp(min=0)
            
            # # Establishing Exact Unlearning Boundary
            # rt_aligned_mask = torch.logical_and(rt_logit > 0.5, rt_logit < 0.7)
            # rt_policy_chosen_logps = rt_logit[rt_aligned_mask][:sum(~ft_aligned_mask)]
            # A = policy_rejected_logps.unsqueeze(0)
            # B = rt_policy_chosen_logps.unsqueeze(0)
            # cos_similarity = F.cosine_similarity(A, B)

            
            # gama = 5
            # chosen_rewards = (policy_chosen_logps - reference_chosen_logps)
            # chosen_losses = 1 - F.sigmoid(gama * (chosen_rewards - z0))
            # rejected_rewards = (policy_rejected_logps - reference_rejected_logps)
            # rejected_losses = 1 - F.sigmoid(gama * (z0 - rejected_rewards))
            # desirable_weight = 1
            # undesirable_weight = 1
            # loss_KTO = torch.mean(torch.cat((desirable_weight * chosen_losses, undesirable_weight * rejected_losses), 0))
            # loss = loss_KTO + GD_loss + 0.6 * TE + 0.3 * cos_similarity
            # print("\ncos_similarity: " + str(cos_similarity) + "======loss_KTO: " + str(loss_KTO) + "======TE_loss: " + str(TE))
            
            # import matplotlib.pyplot as plt
            # import random
            # num_samples = sum(data.df_mask)
            # if epoch in [0, 10, 100, 200, 250, 300]:
            #     plt.figure(figsize=(10, 6))
            #     sampled_rt_logit = random.sample(list(rt_logit.detach().cpu().numpy()), num_samples)
            #     plt.hist(sampled_rt_logit, bins=50, alpha=0.7, color='blue', label='rt_logit')
            #     plt.hist(logits.detach().cpu().numpy(), bins=50, alpha=0.7, color='green', label='logits')
            #     plt.title('Distribution of rt_logit and logits')
            #     plt.xlabel('Value')
            #     plt.ylabel('Frequency')
            #     plt.legend()
            #     plt.tight_layout()
            #     # plt.show()
            #     plt.savefig(str(epoch) + '_contrast_dist-KTO.pdf')
            
            
            loss.backward()
            # for name, param in model.named_parameters():
            #     if param.grad is not None:  
            #         print(f"Layer: {name} | Gradient Norm: {param.grad.norm().item()}")
            #     else:
            #         print(f"Layer: {name} | Gradient is None")
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()

            end_time = time.time()
            epoch_time = end_time - start_time
            
            step_log = {
                'Epoch': epoch,
                'train_loss': loss.item(),
                'train_time': epoch_time
            }
            wandb_log(step_log)
            msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in step_log.items()]
            tqdm.write(' | '.join(msg))

            if (epoch + 1) % self.args.valid_freq == 0:
                valid_loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, valid_log = self.eval(model, data, 'val')
                valid_log['Epoch'] = epoch
                
                wandb_log(valid_log)
                msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in valid_log.items()]
                tqdm.write(' | '.join(msg))
                self.trainer_log['log'].append(valid_log)
                
                # log
                ResLogFile1 = os.path.join('logs', args.dataset) + "_" + args.unlearning_model + "_" + args.df + "_" + str(args.df_size) + ".txt"
                with open(ResLogFile1, "a+") as file:
                    strLog = ' | '.join(msg)
                    file.write(strLog +"\n")  
                
                dist_ckpt = {
                            'dist_model_state': model.state_dict(),
                            'dist_optimizer_state': optimizer.state_dict(),
                        }
                torch.save(dist_ckpt, os.path.join(args.checkpoint_dir, 'model_' + str(epoch) + '_only_retain.pt'))    
                        

                if dt_auc + df_auc > best_metric:
                    best_metric = dt_auc + df_auc
                    best_epoch = epoch

                    print(f'Save best checkpoint at epoch {epoch:04d}. Valid loss = {valid_loss:.4f}')
                    ckpt = {
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                    }
                    torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))
                    torch.save(z, os.path.join(args.checkpoint_dir, 'node_embeddings.pt'))
        
        self.trainer_log['training_time'] = time.time() - start_time

        # Save
        ckpt = {
            'model_state': {k: v.cpu() for k, v in model.state_dict().items()},
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))
        torch.save(z, os.path.join(args.checkpoint_dir, 'node_embeddings.pt'))

    def train_minibatch(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        best_metric = 0

        # MI Attack before unlearning
        if attack_model_all is not None:
            mi_logit_all_before, mi_sucrate_all_before = member_infer_attack(model, attack_model_all, data)
            self.trainer_log['mi_logit_all_before'] = mi_logit_all_before
            self.trainer_log['mi_sucrate_all_before'] = mi_sucrate_all_before
        if attack_model_sub is not None:
            mi_logit_sub_before, mi_sucrate_sub_before = member_infer_attack(model, attack_model_sub, data)
            self.trainer_log['mi_logit_sub_before'] = mi_logit_sub_before
            self.trainer_log['mi_sucrate_sub_before'] = mi_sucrate_sub_before

        data.edge_index = data.train_pos_edge_index
        data.node_id = torch.arange(data.x.shape[0])
        loader = GraphSAINTRandomWalkSampler(
            data, batch_size=args.batch_size, walk_length=2, num_steps=args.num_steps,
        )
        for epoch in trange(args.epochs, desc='Unlearning'):
            model.train()

            epoch_loss = 0
            epoch_time = 0
            for step, batch in enumerate(tqdm(loader, leave=False)):
                start_time = time.time()
                batch = batch.to(device)

                z = model(batch.x, batch.edge_index[:, batch.dr_mask])
                
                # Positive and negative sample
                neg_edge_index = negative_sampling(
                    edge_index=batch.edge_index[:, batch.df_mask],
                    num_nodes=z.size(0))

                logits = model.decode(z, batch.edge_index[:, batch.df_mask])
                label = torch.ones_like(logits, dtype=torch.float, device=device)
                loss = -F.binary_cross_entropy_with_logits(logits, label)

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()

                end_time = time.time()
                epoch_loss += loss.item()
                epoch_time += end_time - start_time

            epoch_loss /= step
            epoch_time /= step

            if (epoch+1) % args.valid_freq == 0:
                valid_loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, valid_log = self.eval(model, data, 'val')

                train_log = {
                    'epoch': epoch,
                    'train_loss': epoch_loss / step,
                    'train_time': epoch_time / step,
                }
                
                for log in [train_log, valid_log]:
                    wandb_log(log)
                    msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
                    tqdm.write(' | '.join(msg))
                    self.trainer_log['log'].append(log)

                if dt_auc + df_auc > best_metric:
                    best_metric = dt_auc + df_auc
                    best_epoch = epoch

                    print(f'Save best checkpoint at epoch {epoch:04d}. Valid loss = {valid_loss:.4f}')
                    ckpt = {
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                    }
                    torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))
                    torch.save(z, os.path.join(args.checkpoint_dir, 'node_embeddings.pt'))

        # Save
        ckpt = {
            'model_state': {k: v.to('cpu') for k, v in model.state_dict().items()},
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_final.pt'))

class KGGradientAscentTrainer(KGTrainer):
    pass