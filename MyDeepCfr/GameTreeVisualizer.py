import networkx as nx
import matplotlib.pyplot as plt
import uuid
import os
import torch
import numpy as np

class GameTreeVisualizer:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_color='lightblue'


    def add_node(self, depth, player, action_taken=None, reward=None):
        node_id = str(uuid.uuid4())
        label = f"d{depth}|P{player}"
        if reward is not None:
            label += f"\nR={reward:.2f}"
        
        cur_color=self.node_color
        if depth==0:
            cur_color='red'
        self.graph.add_node(node_id, label=label,color=cur_color)
        return node_id

    def add_edge(self, parent_id, child_id, action_label=None):
        if action_label is not None:
            self.graph.add_edge(parent_id, child_id, label=f"a{action_label}")
        else:
            self.graph.add_edge(parent_id, child_id)

    def draw(self, figsize=(16, 10), title="Game Tree", save_path=None):
 
        pos = nx.spring_layout(self.graph)

        node_labels = nx.get_node_attributes(self.graph, 'label')
        edge_labels = nx.get_edge_attributes(self.graph, 'label')
        node_colors = [self.graph.nodes[n].get('color', 'lightblue') for n in self.graph.nodes()]

        plt.figure(figsize=figsize)
        nx.spring_layout(self.graph, k=0.5, iterations=20)
        nx.draw(self.graph, pos, with_labels=True, labels=node_labels,
                node_size=800, node_color=node_colors, font_size=6, font_weight='bold', edgecolors='black')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=6)
        plt.title(title, fontsize=10)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    def reset(self):
        self.graph.clear()

    def traverse_full_tree(self, state, env_wrapper, depth=0, parent_node=None, action_taken=None):
        current_player = state["current_player"]

        node_id = self.add_node(depth, current_player, action_taken)
        if parent_node is not None:
            self.add_edge(parent_node, node_id, action_label=action_taken)

        for action in env_wrapper.legal_actions():
            _obs, _rew_for_all, _done, _info = env_wrapper.step(action)
            if _done:
                leaf_id = self.add_node(depth + 1, current_player, action_taken=action, reward=_rew_for_all[current_player])
                self.add_edge(node_id, leaf_id, action_label=action)
            else:
                cur_state = env_wrapper.state_dict()
                self.traverse_full_tree(cur_state, env_wrapper, depth + 1,
                                        parent_node=node_id, action_taken=action)
            env_wrapper.load_state_dict(state)
