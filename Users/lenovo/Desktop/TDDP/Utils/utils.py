import torch
# a function for calculating model parameters

def count_svm_parameters(model):
    """
    Count effective parameters of an sklearn SVC (RBF).
    Parameters = support vectors + dual coefficients + intercepts
    """
    n_support_vectors = model.support_vectors_.size
    n_dual_coef = model.dual_coef_.size
    n_intercepts = model.intercept_.size
    total_params = n_support_vectors + n_dual_coef + n_intercepts
    
    return {
        "n_support_vectors": n_support_vectors,
        "n_dual_coef": n_dual_coef,
        "n_intercepts": n_intercepts,
        "total_params": total_params
    } 
# for calculating Deep Learning models parameters



def count_dl_parameters(model):
    """
    Count total and trainable parameters of a PyTorch model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params
    }