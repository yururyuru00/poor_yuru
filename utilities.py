import random
import torch
import networkx as nx
import numpy as np
from torch_geometric.data import Data


class ExtractSubstructureContextPair:
    def __init__(self, l1):
        self.l1 = l1

    def __call__(self, data):
        num_nodes = data.x.size()[0]
        G = graph_data_obj_to_nx(data)

        center_node_idx = random.choice(range(num_nodes))
        data.x_substruct = data.x
        data.edge_index_substruct = data.edge_index
        data.center_substruct_idx = center_node_idx

        # Get context that is between l1 and the max diameter of the PPI graph
        l1_node_idxes = nx.single_source_shortest_path_length(G, center_node_idx,
                                                              self.l1).keys()
        # l2_node_idxes = nx.single_source_shortest_path_length(G, center_node_idx,
        #                                                       self.l2).keys()
        l2_node_idxes = range(num_nodes)
        context_node_idxes = set(l1_node_idxes).symmetric_difference(
            set(l2_node_idxes))
        if len(context_node_idxes) > 0:
            context_G = G.subgraph(context_node_idxes)
            context_G, context_node_map = reset_idxes(context_G)  # need to
            # reset node idx to 0 -> num_nodes - 1, other data obj does not
            # make sense
            context_data = nx_to_graph_data_obj(context_G, 0)   # use a dummy
            # center node idx
            data.x_context = context_data.x
            data.edge_index_context = context_data.edge_index

        # Get indices of overlapping nodes between substruct and context,
        context_substruct_overlap_idxes = list(context_node_idxes)
        if len(context_substruct_overlap_idxes) > 0:
            context_substruct_overlap_idxes_reorder = [context_node_map[old_idx]
                                                       for old_idx in context_substruct_overlap_idxes]
            data.overlap_context_substruct_idx = \
                torch.tensor(context_substruct_overlap_idxes_reorder)

        return data


def nx_to_graph_data_obj(g, center_id, allowable_features_downstream=None,
                         allowable_features_pretrain=None,
                         node_id_to_go_labels=None):

    # nodes
    n_nodes = g.number_of_nodes()
    n_attributes = len(g.nodes[0])
    x = np.zeros((n_nodes, n_attributes))
    for i in range(n_nodes):
        x[i] = np.array([attribute for attribute in g.nodes[i].values()])
    x = torch.tensor(x, dtype=torch.long)

    nx_node_ids = [n_i for n_i in g.nodes()]
    center_node_idx = nx_node_ids.index(center_id)
    center_node_idx = torch.tensor([center_node_idx], dtype=torch.long)

    # edges
    edges_list = []
    edge_features_list = []
    for node_1, node_2, attr_dict in g.edges(data=True):
        edge_feature = [attr_dict['w1'], attr_dict['w2'], attr_dict['w3'],
                        attr_dict['w4'], attr_dict['w5'], attr_dict['w6'],
                        attr_dict['w7'], 0, 0]  # last 2 indicate self-loop
        # and masking
        edge_feature = np.array(edge_feature, dtype=int)
        # convert nx node ids to data obj node index
        i = nx_node_ids.index(node_1)
        j = nx_node_ids.index(node_2)
        edges_list.append((i, j))
        edge_features_list.append(edge_feature)
        edges_list.append((j, i))
        edge_features_list.append(edge_feature)

    # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
    edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

    # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
    edge_attr = torch.tensor(np.array(edge_features_list),
                             dtype=torch.float)

    try:
        # nx node id is of the form:
        species_id = int(nx_node_ids[0].split('.')[0])
        # species_id.protein_id
        species_id = torch.tensor([species_id], dtype=torch.long)
    except:  # occurs when nx node id has no species id info. For the extract
        # substructure context pair transform, where we convert a data obj to
        # a nx graph obj (which does not have original node id info)
        species_id = torch.tensor([0], dtype=torch.long)    # dummy species
        # id is 0

    # construct data obj
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.species_id = species_id
    data.center_node_idx = center_node_idx

    if node_id_to_go_labels:  # supervised case with go node labels
        # Construct a dim n_pretrain_go_classes tensor and a
        # n_downstream_go_classes tensor for the center node. 0 is no data
        # or negative, 1 is positive.
        downstream_go_node_feature = [0] * len(allowable_features_downstream)
        pretrain_go_node_feature = [0] * len(allowable_features_pretrain)
        if center_id in node_id_to_go_labels:
            go_labels = node_id_to_go_labels[center_id]
            # get indices of allowable_features_downstream that match with elements
            # in go_labels
            _, node_feature_indices, _ = np.intersect1d(
                allowable_features_downstream, go_labels, return_indices=True)
            for idx in node_feature_indices:
                downstream_go_node_feature[idx] = 1
            # get indices of allowable_features_pretrain that match with
            # elements in go_labels
            _, node_feature_indices, _ = np.intersect1d(
                allowable_features_pretrain, go_labels, return_indices=True)
            for idx in node_feature_indices:
                pretrain_go_node_feature[idx] = 1
        data.go_target_downstream = torch.tensor(np.array(downstream_go_node_feature),
                                                 dtype=torch.long)
        data.go_target_pretrain = torch.tensor(np.array(pretrain_go_node_feature),
                                               dtype=torch.long)

    return data


def graph_data_obj_to_nx(data):

    G = nx.Graph()

    x = data.x.cpu().numpy()
    for idx, attributes in enumerate(x):
        G.add_node(idx)
        attr = {idx: {dim_idx: attribute for dim_idx,
                      attribute in enumerate(attributes)}}
        nx.set_node_attributes(G, attr)

    edge_index = data.edge_index.cpu().numpy()
    n_edges = edge_index.shape[1]
    for j in range(0, n_edges, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        if not G.has_edge(begin_idx, end_idx):
            G.add_edge(begin_idx, end_idx)

    return


def reset_idxes(G):
    """
    Resets node indices such that they are numbered from 0 to num_nodes - 1
    :param G:
    :return: copy of G with relabelled node indices, mapping
    """
    mapping = {}
    for new_idx, old_idx in enumerate(G.nodes()):
        mapping[old_idx] = new_idx
    new_G = nx.relabel_nodes(G, mapping, copy=True)
    return new_G, mapping
