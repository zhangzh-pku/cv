import numpy as np
import cv2
from torch import digamma

# eps may help you to deal with numerical problem
eps = 1e-5


def bn_forward_test(x, gamma, beta, mean, var):

    #----------------TODO------------------
    # Implement forward
    #----------------TODO------------------
    x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)
    out = gamma * x_hat + beta
    return out


def bn_forward_train(x, gamma, beta):

    #----------------TODO------------------
    # Implement forward
    sample_mean = np.mean(x, axis=0)
    sample_var = np.var(x, axis=0)
    x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)
    out = gamma * x_hat + beta
    #----------------TODO------------------

    # save intermidiate variables for computing the gradient when backward

    cache = (gamma, x, sample_mean, sample_var, x_hat)
    return out, cache


def bn_backward(dout, cache):

    #----------------TODO------------------
    # Implement backward
    #----------------TODO------------------
    gamma, x, sample_mean, sample_var, x_hat = cache
    N = dout.shape[0]
    dgamma = np.sum(x_hat * dout, axis=0)
    dbeta = np.sum(dout, axis=0)

    dx_hat = dout * gamma
    dx = N * dx_hat - np.sum(dx_hat,
                             axis=0) - x_hat * np.sum(dx_hat * x_hat, axis=0)
    dx *= (1.0 / N) / np.sqrt(sample_var + eps)
    return dx, dgamma, dbeta


# This function may help you to check your code
def print_info(x):
    print('mean:', np.mean(x, axis=0))
    print('var:', np.var(x, axis=0))
    print('------------------')
    return


if __name__ == "__main__":

    # input data
    train_data = np.zeros((9, 784))
    #抱歉助教我这里实在不明白为什么cv2不用绝对路径他就读不进来数据TAT
    _path = "/Users/zhangzhihao/Library/Mobile Documents/com~apple~CloudDocs/data_science/3_down/cv/02_assignment/02_assignment/batch_normalization/mnist_subset/"
    for i in range(9):
        path = _path + str(i) + ".png"
        train_data[i, :] = cv2.imread(path,
                                      cv2.IMREAD_GRAYSCALE).reshape(-1) / 255.

    gt_y = np.zeros((9, 1))
    gt_y[0] = 1

    val_data = np.zeros((1, 784))
    path = _path + "/9.png"
    #img = np.fromfile(path, dtype=np.uint8)
    val_data[0, :] = cv2.imread(path, cv2.IMREAD_GRAYSCALE).reshape(-1) / 255.
    val_gt = np.zeros((1, 1))

    np.random.seed(14)

    # Intialize MLP  (784 -> 16 -> 1)
    MLP_layer_1 = np.random.randn(784, 16)
    MLP_layer_2 = np.random.randn(16, 1)

    # Initialize gamma and beta
    gamma = np.random.randn(16)
    beta = np.random.randn(16)

    lr = 1e-1
    loss_list = []

    # ---------------- TODO -------------------
    # compute mean and var for testing
    # add codes anywhere as you need
    # ---------------- TODO -------------------
    mean = np.mean(train_data.dot(MLP_layer_1), axis=0)
    var = np.var(train_data.dot(MLP_layer_1), axis=0)
    momentums = 0.9
    # training
    for i in range(50):
        # Forward
        output_layer_1 = train_data.dot(MLP_layer_1)
        output_layer_1_bn, cache = bn_forward_train(output_layer_1, gamma,
                                                    beta)
        output_layer_1_act = 1 / (1 + np.exp(-output_layer_1_bn)
                                  )  #sigmoid activation function
        output_layer_2 = output_layer_1_act.dot(MLP_layer_2)
        pred_y = 1 / (1 + np.exp(-output_layer_2)
                      )  #sigmoid activation function

        # compute loss
        loss = -(gt_y * np.log(pred_y) + (1 - gt_y) * np.log(1 - pred_y)).sum()
        print("iteration: %d, loss: %f" % (i + 1, loss))
        loss_list.append(loss)

        # Backward : compute the gradient of paratmerters of layer1 (grad_layer_1) and layer2 (grad_layer_2)
        grad_pred_y = -(gt_y / pred_y) + (1 - gt_y) / (1 - pred_y)
        grad_activation_func = grad_pred_y * pred_y * (1 - pred_y)
        grad_layer_2 = output_layer_1_act.T.dot(grad_activation_func)
        grad_output_layer_1_act = grad_activation_func.dot(MLP_layer_2.T)
        grad_output_layer_1_bn = grad_output_layer_1_act * (
            1 - output_layer_1_act) * output_layer_1_act
        grad_output_layer_1, grad_gamma, grad_beta = bn_backward(
            grad_output_layer_1_bn, cache)
        grad_layer_1 = train_data.T.dot(grad_output_layer_1)

        #compute running mean
        gamma, x, sample_mean, sample_var, x_hat = cache
        mean = mean * momentums + sample_mean * (1 - momentums)
        var = var * momentums + sample_var * (1 - momentums)

        # update parameters
        gamma -= lr * grad_gamma
        beta -= lr * grad_beta
        MLP_layer_1 -= lr * grad_layer_1
        MLP_layer_2 -= lr * grad_layer_2

    # validate
    output_layer_1 = val_data.dot(MLP_layer_1)
    output_layer_1_bn = bn_forward_test(output_layer_1, gamma, beta, mean, var)
    output_layer_1_act = 1 / (1 + np.exp(-output_layer_1_bn)
                              )  #sigmoid activation function
    output_layer_2 = output_layer_1_act.dot(MLP_layer_2)
    pred_y = 1 / (1 + np.exp(-output_layer_2))  #sigmoid activation function
    loss = -(val_gt * np.log(pred_y) + (1 - val_gt) * np.log(1 - pred_y)).sum()
    print("validation loss: %f" % (loss))
    loss_list.append(loss)
    _path = "/Users/zhangzhihao/Library/Mobile Documents/com~apple~CloudDocs/data_science/3_down/cv/02_assignment/02_assignment"
    np.savetxt(_path + "/results/bn_loss.txt", loss_list)
