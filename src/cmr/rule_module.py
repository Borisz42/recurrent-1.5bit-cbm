import torch
from src.cmr.logic import reasoning

class RuleModule(torch.nn.Module):
    def __init__(self, rule_emb_size, n_tasks, n_rules):
        """
        Abstract class for rule modules. A rule module stores rule embeddings, and provides:
        1. A way to decode them into polarities and relevances.
        2. A way to predict the task given the symbolic rules.
        3. A way to compute the 'concept reconstruction'.
        """
        super().__init__()
        self.rules = torch.nn.Embedding(n_tasks * n_rules, rule_emb_size)
        self.n_rules = n_rules
        self.rule_emb_size = rule_emb_size

    def copy_embedding(self, task_idx, rule_idx_from, rule_idx_to):
        embedding_from = self.rules.weight[task_idx * self.n_rules + rule_idx_from]
        self.rules.weight.data[task_idx * self.n_rules + rule_idx_to] = torch.clone(embedding_from)

    def decode_rules(self, rule_embs):
        """
        Returns a tensor ending on dimensions (concepts, 3) where the last dimension is the probability for
        positive polarity, negative polarity and irrelevance for a specific concept.
        """
        raise NotImplementedError

    def calc_y(self, c, pospolarity, relevance):
        return reasoning(self.logic, c, pospolarity, relevance)

    def calc_c_rec(self, pospolarity, relevance):
        raise NotImplementedError


class ProbRDCat(RuleModule):
    def __init__(self, rule_emb_size, n_concepts, n_tasks, n_rules):
        """
        A rule module where rules are decoded into a categorical variable defining positive polarity, negative polarity,
        and irrelevance. Therefore, they are mutually exclusive.
        """
        super().__init__(rule_emb_size, n_tasks, n_rules)
        self.logic = None
        self.n_concepts = n_concepts
        self.rule_emb_size = rule_emb_size
        self.rule_decoder = torch.nn.Sequential(
            torch.nn.Linear(self.rule_emb_size, self.rule_emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.rule_emb_size, 3 * self.n_concepts),
        )

    def decode_rules(self, rule_embs):
        shape = rule_embs.shape[:-1]  # keep all dims except the embedding dim
        shape += (self.n_concepts, 3)
        logits = self.rule_decoder(rule_embs).view(shape)
        return torch.softmax(logits, dim=-1)

    def calc_c_rec(self, pospolarity, relevance):
        return 0.5 * (1 - relevance) + pospolarity
