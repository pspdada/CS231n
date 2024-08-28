from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]
    for i in range(num_train):
        scores = X[i].dot(W)  # (1, D) * (D, C) = (1, C)
        # 避免数值下溢: 为了避免指数运算导致的数值下溢问题，对得分进行偏移，
        # 减去该样本得分中的最大值。这不影响softmax函数的相对概率分布，但有助于保持数值稳定性
        scores -= np.max(scores)
        exp_scores = np.exp(scores)
        sum_exp_scores = np.sum(exp_scores)
        # 交叉熵损失是对数似然损失的负值，衡量预测概率分布与实际标签分布的差异。
        loss += -np.log(exp_scores[y[i]] / sum_exp_scores)
        for j in range(num_classes):
            p_j = exp_scores[j] / sum_exp_scores
            dW[:, j] += X[i] * p_j
        dW[:, y[i]] -= X[i]  # 正确类别的梯度要减去样本

    loss /= num_train
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW += 2 * reg * W

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    scores = X.dot(W)  # (N, D) * (D, C) = (N, C)
    scores -= np.max(scores, axis=1, keepdims=True)  # 求每行的最大值
    exp_scores = np.exp(scores)
    sum_exp_scores = np.sum(exp_scores, axis=1, keepdims=True)  # 求每行的和，（N, 1）
    # nparray 的高级索引，从ndarray对象的每一行中选择指定列的一个元素。因此，行索引包含所有行号，列索引指定要选择的元素。
    arrange = np.arange(num_train)
    p_y_i = exp_scores[arrange, y] / np.squeeze(sum_exp_scores)  # (N, )
    loss = np.sum(-np.log(p_y_i))
    loss /= num_train
    loss += reg * np.sum(W * W)

    p = exp_scores / sum_exp_scores
    p[np.arange(num_train), y] -= 1
    dW = X.T @ p / num_train
    dW += 2 * reg * W

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
