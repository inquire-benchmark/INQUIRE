"""This file contains utility functions to assist with retrieval and reranking evaluation."""

import numpy as np
import sklearn.metrics

class MetricAverage():
    """This is a helper to keep track of multiple metric averages simultaneously."""
    def __init__(self):
        self.sum = None
        self.avg = None
        self.count = 0
        
    def update(self, val):
        if type(val) == list or type(val) == tuple:
            val = np.asarray(val)
        
        # Initialization, if no previous data was seen
        if self.sum is None:
            self.sum = np.zeros_like(val)
            self.avg = np.zeros_like(val)
            
        self.count += 1
        self.sum += val
        self.avg = self.sum / self.count


def ndcg_at_k(y_true, y_pred, count_pos, k):
    """Compute nDCG at k.
    
    nDCG is a bit more complex to get, since it assumes we have access to the
    labels and predictions for the full dataset. Since we only get top k, we
    need to append pseudo labels for all the remaining unknown positives to
    get an accurate number. In this case, we set the "predicted" score of 
    these unknown positives to a very low number, as if they are ranked last.
    """
    
    num_pos_unknown = count_pos - sum(y_true)
    y_true_expanded = y_true.tolist() + [1]*num_pos_unknown
    y_pred_expanded = y_pred.tolist() + [-100]*num_pos_unknown
    return sklearn.metrics.ndcg_score([y_true_expanded], [y_pred_expanded], k=k)

def ap_at_k(y_true, y_pred, count_pos, k):
    """Compute AP at k for the retrieval task.
    
    Typical implementations of AP normalize by the number of thresholds. However, in the
    retrieval setting this does not account for missed positives. We use a AP with a
    different normalization that accounts for this. For a full treatment of why this
    is done, see the supplementary material of the INQUIRE benchmark paper. """
    unnormalized_ap = sklearn.metrics.average_precision_score(y_true[:k], y_pred[:k]) if sum(y_true[:k]) > 0 else 0
    ap = unnormalized_ap *  y_true[:k].sum() / np.amin([count_pos, k])
    return ap


def mrr(y_true, y_pred, count_pos):
    """Calculate mean reciprocal rank."""
    
    # make sure rank is ordered
    order = np.flip(np.argsort(y_pred))
    
    for i in range(len(y_true)):
        if y_true[order][i] == 1:
            return 1 / (i + 1)
        
    return 0.0

        
def compute_retrieval_metrics(y_true, y_pred, count_pos):
    """Computes complete retrieval metrics given the ground truth and predictions
    of top k retrievals. The total number of positives in the dataset is required
    to compute recall and nDCG.

    Args:
        y_true (np.ndarray): Binary relavance of top k retrievals.
        y_pred (np.ndarray): Predicted relavance score of top k retrievals.
        count_pos (int): Number of total relavant images in the dataset.

    Returns:
        (precision, recall, Average Precision, NDCG, MRR)
    """
    
    k = len(y_true)
    precision = sum(y_true) / k
    recall = sum(y_true) / count_pos
    ap = ap_at_k(y_true, y_pred, count_pos, k)
    ndcg = ndcg_at_k(y_true, y_pred, count_pos, k)
    mrr_score = mrr(y_true, y_pred, count_pos)
    return precision, recall, ap, ndcg, mrr_score
