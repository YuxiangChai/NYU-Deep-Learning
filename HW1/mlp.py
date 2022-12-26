import torch
import copy


class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        # TODO: Implement the forward function
        
        self.cache['x'] = x
        z1 = torch.matmul(x, torch.transpose(self.parameters['W1'], 0, 1)) + self.parameters['b1']
        self.cache['z1'] = z1
        if self.f_function == 'relu':
            z2 = torch.relu(z1)
        elif self.f_function == 'sigmoid':
            z2 = torch.sigmoid(z1)
        else:
            z2 = copy.deepcopy(z1)
        self.cache['z2'] = z2
        z3 = torch.matmul(z2, torch.transpose(self.parameters['W2'], 0, 1)) + self.parameters['b2']
        self.cache['z3'] = z3
        if self.g_function == 'relu':
            y_hat = torch.relu(z3)
        elif self.g_function == 'sigmoid':
            y_hat = torch.sigmoid(z3)
        else:
            y_hat = copy.deepcopy(z3)
        self.cache['y_hat'] = y_hat
        
        return y_hat
    
    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        # TODO: Implement the backward function
        
        if self.g_function == 'relu':
            dy_hatdz3 = (self.cache['z3'] > 0) * 1.0
        elif self.g_function == 'sigmoid':
            dy_hatdz3 = torch.mul(self.cache['y_hat'], (1-self.cache['y_hat']))
        else:
            dy_hatdz3 = torch.ones(self.cache['y_hat'].shape[0], self.cache['y_hat'].shape[1])
        
        if self.f_function == 'relu':
            dz2dz1 = (self.cache['z1'] > 0) * 1.0
        elif self.f_function == 'sigmoid':
            dz2dz1 = torch.mul(self.cache['z2'], (1-self.cache['z2']))
        else:
            dz2dz1 = torch.ones(self.cache['z2'].shape[0], self.cache['z2'].shape[1])
            
        dJdb2 = torch.sum(torch.mul(dJdy_hat, dy_hatdz3), dim=0)
        dJdW2 = torch.matmul(torch.transpose(torch.mul(dJdy_hat, dy_hatdz3), 0, 1), self.cache['z2'])
        dJdb1 = torch.sum(torch.mul(torch.matmul(torch.mul(dJdy_hat, dy_hatdz3), self.parameters['W2']), dz2dz1), dim=0)
        dJdW1 = torch.matmul(torch.transpose(torch.mul(torch.matmul(torch.mul(dJdy_hat, dy_hatdz3), self.parameters['W2']), dz2dz1), 0, 1), self.cache['x'])
        
        self.grads['dJdb2'] = dJdb2
        self.grads['dJdW2'] = dJdW2
        self.grads['dJdb1'] = dJdb1
        self.grads['dJdW1'] = dJdW1

    
    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the mse loss
    B = y.shape[0]
    nf = y.shape[1]
    J = torch.mean(torch.pow((y_hat - y), 2))
    dJdy_hat = 2*(y_hat - y) / (B*nf)

    return J, dJdy_hat

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the bce loss
    B = y.shape[0]
    nf = y.shape[1]
    log_y_hat = torch.log(y_hat)
    log_1_y_hat = torch.log(1 - y_hat)
    
    loss = torch.mean((-1)*(torch.mul(y, log_y_hat) + torch.mul((1-y), log_1_y_hat)))
    dJdy_hat = (- y / y_hat + (1-y) / (1-y_hat)) / (B*nf)
    
    return loss, dJdy_hat











