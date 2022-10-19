import torch
import torch.nn as nn
import torch.nn.functional as fun
import numpy as np
import os.path
from datetime import datetime

class ClassifierNN(nn.Module):
    """
    Provide a neural network model for classification.

    Provide a neural network (NN) model for classification.  The NN is a
    simple, fully connected feed-forward network.  The layout of the NN is
    specified at construction time by providing a tuple.  The length of the
    tuple corresponds to the number of network layers (including input and
    output layers).  Each tuple entry specifies the number of nodes in the
    corresponding layer.  The width of the input and output layer must
    correspond to the number of input variables and classes, respectively.

    The non-linear activation function for the hidden layers is relu.  The
    output activation is linear during training and sigmoid in inference mode.
    We use nn.BCELoss() as the loss function during training, as usual for
    binary classifiers.

    The recommended optimizer is Adam.

    In case you move the classifier to an accelerator (such as a GPU) make sure
    you construct the optimizer after.  Of course, different optimizers and
    loss functions can be used; make sure the implications are understood, in
    particular for the output layer activation (see above).
    """
    def __init__(self, idim,
                 activation=fun.relu):
        super().__init__()

        self.last_save = None
        self.layout = (idim, 256, 128, 1)
        self.inference_mode = True  # training clients: change this attribute to False
        self.activation = activation
        self.layers = nn.ModuleList()
        for num_nodes, num_nodes_next in zip(self.layout[:-1], self.layout[1:]):
            self.layers.append(nn.Linear(num_nodes, num_nodes_next))

    def forward(self, x):
        for layer in self.layers[:-1]:
            if not isinstance(x, torch.Tensor):
                x = torch.Tensor(x)
            x = self.activation(layer(x))

        #x = self.layers[-1](x)
        x = torch.sigmoid(self.layers[-1](x))
        #if self.inference_mode:
        #    x = torch.sigmoid(self.layers[-1](x))
        #else:
        #    x = self.layers[-1](x)
        return x

    def train(self, mode=True):
        super(ClassifierNN, self).train()
        self.inference_mode = False

    def eval(self):
        super(ClassifierNN, self).eval()
        self.inference_mode = True

    def save_weights(self, tag=None, time_stamp=True, directory=None):
        weight_file_path = 'classifier_weights_'
        if tag is not None:
            weight_file_path += '{}_'.format(tag)
        for width in self.layout[:-1]:
            weight_file_path += '{}x'.format(width)
        weight_file_path += '{}'.format(self.layout[-1])
        if time_stamp:
            weight_file_path += '_{}'.format(datetime.now().strftime('%Y%m%d%H%M%S'))
        weight_file_path += '.pt'
        if directory is not None:
            weight_file_path = os.path.join(directory, weight_file_path)

        torch.save(self.state_dict(), weight_file_path)
        self.last_save = weight_file_path

        return weight_file_path

    def train_clf(self, model, train_loader, val_loader):
        grace = 0
        max_epochs=10
        min_gain=0.01
        grace_limit=4
        learning_rate=0.002
        momentum=0.002
        weight_decay=1e-5
        save_dir='/Users/mattia/Desktop/MLaaS4HEP/weights'
        weight_file_tag=None
        train_history = []
        test_history = []
        loss_function = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        model.train()

        weigh_path = None
        best_loss = 1.0
        for epoch in range(max_epochs):
            mean_train_loss = 0.0
            for i, (xs, ys) in enumerate(train_loader):
                optimizer.zero_grad() # reset gradients
                outputs = model(xs)
                train_loss = loss_function(outputs, ys)
                train_loss.backward() # gradient back propagation
                optimizer.step()
                mean_train_loss = (mean_train_loss * i + float(train_loss)) / (i + 1)
            train_history.append(mean_train_loss)

            mean_test_loss = 0.0
            for i, (xs, ys) in enumerate(val_loader):
                outputs = model(xs)
                eval_loss = loss_function(outputs, ys)
                mean_test_loss = (mean_test_loss * i + float(eval_loss)) / (i + 1)
            test_history.append(mean_test_loss)

            print('Epoch {}, mean train/test loss: {:.4f}/{:.4f}'.format(epoch+1, mean_train_loss, mean_test_loss))
            if (best_loss - mean_test_loss) / best_loss < min_gain:
                if grace == 0:
                    print('Entering grace period (limit {})'.format(grace_limit))
                    grace += 1
                elif grace < grace_limit:
                    grace += 1
                else:
                    print('Nothing more to learn. Training finished.')
                    break
            else:
                if grace > 0:
                    grace = 0
                    print('Survived grace period.')
                best_loss = mean_test_loss
                weight_path = model.save_weights(tag=weight_file_tag, time_stamp=False, directory=save_dir)
                print("Saved network parameters to '{}'.".format(weight_path))
        else:
            print('Maximum number of epochs ({}) reached. Training terminated.'.format(max_epochs))

        #return train_history, test_history, weight_path


def model(idim):
    """Simple pyTorch model for testing purpose"""
    torch_model = ClassifierNN(idim)
    return torch_model
