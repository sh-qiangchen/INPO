# Train GNN The first step is to train a GNN model, on either link prediction or node classification
python train_gnn.py --dataset CS
python train_gnn.py --dataset Cora
python train_gnn.py --dataset DBLP
python train_gnn.py --dataset PubMed
python train_gnn.py --dataset ogbl-collab

# Unlearn Then we can delete information from the trained GNN model. Based on what you want to delete, run one of the three scrips
# To unlearn edges, please run
python delete_gnn.py --unlearning_model gnndelete --df_size 0.5
python delete_gnn.py --dataset Cora --unlearning_model gnndelete --df_size 1.0
python delete_gnn.py --dataset PubMed --unlearning_model NPO --df_size 0.5 --df out
python delete_gnn.py --dataset PubMed --unlearning_model gradient_ascent --df_size 0.5 --df out
python delete_gnn.py --dataset PubMed --unlearning_model gnndelete_all --df_size 0.5 --df out 
python delete_gnn.py --dataset DBLP --unlearning_model gnndelete --df_size 0.5 --df out
python delete_gnn.py --dataset ogbl-collab --unlearning_model gnndelete_all --df_size 0.5 --df out 
python delete_gnn.py --dataset DBLP --unlearning_model gnndelete_all --df_size 0.5 --df out --loss_type both_layerwise
python delete_gnn.py --dataset DBLP --unlearning_model gradient_ascent --df_size 0.5 --df out

python delete_gnn.py --dataset DBLP --unlearning_model gnndelete_lora --df_size 0.5 --df out --loss_type both_layerwise


# overall
python delete_gnn.py --dataset Cora --unlearning_model retrain --df_size 0.5 --df out
python delete_gnn.py --dataset Cora --unlearning_model gradient_ascent --df_size 0.5 --df out
python delete_gnn.py --dataset Cora --unlearning_model gif --df_size 0.5 --df out
python delete_gnn.py --dataset Cora --unlearning_model utu --df_size 0.5 --df out
python delete_gnn.py --dataset Cora --unlearning_model DPO --df_size 0.5 --df out
python delete_gnn.py --dataset Cora --unlearning_model NPO --df_size 0.5 --df out
python delete_gnn.py --dataset Cora --unlearning_model gnndelete_all --df_size 0.5 --df out 

python delete_gnn.py --dataset DBLP --unlearning_model retrain --df_size 0.5 --df out
python delete_gnn.py --dataset DBLP --unlearning_model gradient_ascent --df_size 0.5 --df out
python delete_gnn.py --dataset DBLP --unlearning_model gif --df_size 0.5 --df out
python delete_gnn.py --dataset DBLP --unlearning_model gnndelete_all --df_size 0.5 --df out 
python delete_gnn.py --dataset DBLP --unlearning_model utu --df_size 0.5 --df out
python delete_gnn.py --dataset DBLP --unlearning_model DPO --df_size 0.5 --df out
python delete_gnn.py --dataset DBLP --unlearning_model NPO --df_size 0.5 --df out


python delete_gnn.py --dataset PubMed --unlearning_model retrain --df_size 0.5 --df out
python delete_gnn.py --dataset PubMed --unlearning_model gradient_ascent --df_size 0.5 --df out
python delete_gnn.py --dataset PubMed --unlearning_model gif --df_size 0.5 --df out
python delete_gnn.py --dataset PubMed --unlearning_model gnndelete_all --df_size 0.5 --df out 
python delete_gnn.py --dataset PubMed --unlearning_model utu --df_size 0.5 --df out
python delete_gnn.py --dataset PubMed --unlearning_model DPO --df_size 0.5 --df out
python delete_gnn.py --dataset PubMed --unlearning_model NPO --df_size 0.5 --df out

python delete_gnn.py --dataset CS --unlearning_model gif --df_size 0.5 --df out
python delete_gnn.py --dataset CS --unlearning_model utu --df_size 0.5 --df out


python delete_gnn.py --dataset ogbl-collab --unlearning_model retrain --df_size 0.5 --df out
python delete_gnn.py --dataset ogbl-collab --unlearning_model gradient_ascent --df_size 0.5 --df out
python delete_gnn.py --dataset ogbl-collab --unlearning_model gif --df_size 0.5 --df out
python delete_gnn.py --dataset ogbl-collab --unlearning_model gnndelete_all --df_size 0.5 --df out
python delete_gnn.py --dataset ogbl-collab --unlearning_model utu --df_size 0.5 --df out
python delete_gnn.py --dataset ogbl-collab --unlearning_model NPO --df_size 0.5 --df out


python delete_gnn.py --dataset CS --unlearning_model NPO --df_size 0.5 --df out
python delete_gnn.py --dataset CS --unlearning_model gnndelete_all --df_size 0.5 --df out 




python delete_gnn.py --dataset DBLP --unlearning_model gnndelete_all --df_size 0.5 --df out 
python delete_gnn.py --dataset DBLP --unlearning_model utu --df_size 0.5 --df out 

python delete_gnn.py --dataset DBLP --unlearning_model gnndelete_NPO --df_size 0.5 --df out --loss_type both_layerwise


# To unlearn nodes, please run
python delete_nodes.py

# To unlearn node features, please run
python delete_node_feature.py



python delete_gnn.py --dataset Cora --unlearning_model KTO --df_size 0.5 --df out