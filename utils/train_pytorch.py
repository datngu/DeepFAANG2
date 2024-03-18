import os
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import math


 
## an often use function

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




## accuracies metric functions

def binary_AUC(logits, labels, activation=torch.sigmoid):
    """
    Compute the Area Under the Receiver Operating Characteristic Curve (AUC-ROC) for binary classification.

    Parameters:
        - logits (torch.Tensor): Model logits or predicted scores (output before activation function).
        - labels (torch.Tensor): True binary labels (0 or 1).
        - activation (torch.nn.functional, optional): Activation function applied to logits (default is torch.sigmoid).

    Returns:
        - float: AUC-ROC score.

    Example:
        logits = torch.tensor([0.2, 0.7, 0.4, 0.9])
        labels = torch.tensor([0, 1, 0, 1])
        auc = binary_AUC(logits, labels, activation=torch.sigmoid)
        print("AUC-ROC Score:", auc)
    """

    # Ensure logits and labels are on the CPU
    logits, labels = logits.detach().cpu(), labels.cpu()
    # Apply activation function to logits if provided
    if activation is not None:
        predicts = activation(logits)
    else:
        predicts = logits

    # Reshape tensors and convert to NumPy arrays
    labels = labels.numpy().flatten()
    predicts = predicts.numpy().flatten()
    # Calculate AUC-ROC using sklearn's roc_auc_score
    auc = roc_auc_score(labels, predicts)
    # Return the AUC-ROC score as a float
    return auc.item()


def binary_accuracy(logits, labels, cutoff=0):
    """
    Calculate binary accuracy given model logits and true labels.

    Parameters:
        - logits (torch.Tensor): Model logits or predicted scores (output before activation function).
        - labels (torch.Tensor): True binary labels (0 or 1).
        - cutoff (float, optional): Threshold for binary classification (default is 0).

    Returns:
        - float: Binary accuracy.

    Example:
        logits = torch.tensor([0.2, 0.7, 0.4, 0.9])
        labels = torch.tensor([0, 1, 0, 1])
        accuracy = binary_accuracy(logits, labels, cutoff=0.5)
        print("Binary Accuracy:", accuracy)
    """

    # Ensure logits and labels are on the CPU
    logits, labels = logits.cpu(), labels.cpu()
    # Convert logits to binary predictions using the specified cutoff
    predicts = (logits > cutoff).float()
    # Calculate binary accuracy
    accuracy = (predicts == labels).float().mean()
    # Return the binary accuracy as a float
    return accuracy.item()



def multiple_class_accuracy(logits, labels):
    """
    Compute multiple classification accuracy score.

    Parameters:
        - logits (torch.Tensor): Model logits or predicted scores (output before softmax).
        - labels (torch.Tensor): True class labels.

    Returns:
        - float: Accuracy score.

    Example:
        logits = torch.tensor([[0.2, 0.7, 0.1], [0.8, 0.1, 0.1], [0.3, 0.4, 0.3]])
        labels = torch.tensor([1, 0, 2])
        accuracy = multiple_class_accuracy(logits, labels)
        print("Accuracy:", accuracy)
    """

    # Ensure logits and labels are on the CPU
    logits, labels = logits.cpu(), labels.cpu()
    # Convert logits to predicted class indices
    predicts = logits.argmax(-1).float()
    # Calculate accuracy
    acc = (predicts == labels).float().mean()
    # Return the accuracy as a float
    return acc.item()


def regression_r2(logits, labels):
    """
    Compute the R-squared value (coefficient of determination) between logits and labels.

    Parameters:
        - logits (torch.Tensor): Model logits or predicted values.
        - labels (torch.Tensor): True labels of the data.

    Returns:
        - float: R-squared value.
    """

    # Ensure logits and labels are on the CPU
    x, y = logits.cpu(), labels.cpu()
    # Calculate mean of x and y
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    # Calculate covariance and variances
    cov_xy = torch.sum((x - mean_x) * (y - mean_y))
    var_x = torch.sum((x - mean_x)**2)
    var_y = torch.sum((y - mean_y)**2)
    # Calculate Pearson correlation coefficient (r)
    r = cov_xy / torch.sqrt(var_x * var_y)
    # Calculate R-squared value
    r2 = r**2
    # Return the R-squared value as a float
    return r2.item()



####################################

class EarlyStopper:
    """
    A simple implementation of EarlyStopper.
    """
    def __init__(self, patience=5, min_delta=0.0):
        """
        Initialize a new instance of EarlyStopper.

        Args:
            patience (int): Maximum number of epoch to wait for validation loss improvement.
            min_delta (float): Tolorent cutoff.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        """
        Update validation_loss.
            return 'True' when maximum patience is reached.
            return 'False' otherwise. 

        Args:
            validation_loss (float): validation loss of the current epoch.
        """

        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True

        ## check nan loss: no more training!
        if math.isnan(validation_loss):
            return True
        else:
            return False


class Trainer(object):
    """
    A simple implementation of PyTorch trainer to simplify model development.
    """
    def __init__(self, 
        model, 
        criterion, 
        optimizer, 
        metric_function = None,
        num_epochs = 100, 
        early_stoper = 5
        ):
        """
        Initialize a new instance of Trainer.

        Args:
            model: PyTorch model class.
            criterion: PyToch loss function class.
            optimizer: PyToch optimizer class.
            metric_function: a function to compute accuracy score, it can be any function as long as it can take 'logits' and 'labels' as input and output an accuracy score.
            num_epochs: maximum epochs to train the model
            early_stoper: maximum number of epoch to wait for validation loss improvement.
        """
        device = next(model.parameters()).device
        print(f'Using model device: {device}')
        self.device = device
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.metric_function = metric_function
        self.num_epochs = num_epochs
        self.early_stoper = EarlyStopper(early_stoper)
        self.epoch = 0
        self.train_stat = [['epoch', 'loss', 'score', 'is_improve']]
        self.val_stat = [['epoch', 'loss', 'score', 'is_improve']]
        
        ## load require package
        #import os
        #from tqdm import tqdm


    def _train_epoch(self, data_loader):
        """
        Internal method to train an epoch.
        return:
            aggregated_loss: mean loss over all mini batches.
            aggregated_score: mean accuracy score over all mini batches, computed by the given metric_function.

        Args:
            data_loader: a data_loader object to generate inputs, and labels for training
        """
        self.model.train()
        running_loss = 0.00
        running_score = 0.00
        current_score = 0.0
        aggregated_score = 0.00

        progress_bar = tqdm(enumerate(data_loader), total = len(data_loader), position = 0, leave = True)
        for idx, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            logits = self.model(inputs)
            ## make sure no dimension error for classification tasks
            logits = torch.squeeze(logits, -1) 
            loss = self.criterion(logits, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            aggregated_loss =  running_loss/(idx+1)

            if self.metric_function is not None:
                # handle potention error caused by the metric_function
                try:
                    current_score = self.metric_function(logits, labels)
                except:
                    current_score = aggregated_score

                running_score += current_score
                aggregated_score = running_score/(idx+1)

            progress_bar.set_description(f"Training [{self.epoch}/{self.num_epochs}], loss [{loss.item():.4f}/{aggregated_loss:.4f}], Score [{current_score:.4f}/{aggregated_score:.4f}]")
        
        return aggregated_loss, aggregated_score



    def _val_epoch(self, data_loader):
        """
        Internal method to validate an epoch.
        return:
            aggregated_loss: mean loss over all mini batches.
            aggregated_score: mean accuracy score over all mini batches, computed by the given metric_function.

        Args:
            data_loader: a data_loader object to generate inputs, and labels for validation
        """
        self.model.eval()
        running_loss = 0.00
        running_score = 0.00
        current_score = 0.0
        aggregated_score = 0.00
        progress_bar = tqdm(enumerate(data_loader), total = len(data_loader), position = 0, leave = True)
        for idx, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                logits = self.model(inputs)
            
            ## make sure no dimension error for classification tasks
            logits = torch.squeeze(logits, -1) 
            
            loss = self.criterion(logits, labels)
            running_loss += loss.item()
            aggregated_loss =  running_loss/(idx+1)

            if self.metric_function is not None:
                # handle potention error caused by the metric_function
                try:
                    current_score = self.metric_function(logits, labels)
                except:
                    current_score = aggregated_score

                running_score += current_score
                aggregated_score = running_score/(idx+1)

            
            progress_bar.set_description(f"Validation [{self.epoch}/{self.num_epochs}], loss [{loss.item():.4f}/{aggregated_loss:.4f}], Score [{current_score:.4f}/{aggregated_score:.4f}]")

        return aggregated_loss, aggregated_score

    
    @staticmethod
    def _list_to_csv(file_name, my_list):
        with open(file_name, mode='w') as file:
            for line in my_list:
                line = [str(i) for i in line]
                line = ','.join(line)
                file.write(line + '\n')


    def fit(self, train_loader, val_loader_fw, val_loader_rc, output = './model'):
        """
        The main method to fit the model.
        return:
            'output' directory contains:
                model checkpoints for each epoch
                the best model based on validation loss.
                train_log.csv and val_log.csv.

        Args:
            train_loader: a data_loader object to generate inputs, and labels for training.
            val_loader: a data_loader object to generate inputs, and labels for validation.
            output: output directory.
        """
        if not os.path.exists(output):
            os.makedirs(output)

        for epoch in range(1, self.num_epochs + 1):
            self.epoch = epoch
            is_improve = False
            
            train_los, train_score = self._train_epoch(train_loader)
            # print(f'Finished training epoch [{epoch}/{self.num_epochs}], mean loss: {train_los}, mean score: {train_score}')
            

            ## do validation for both fw and rc strand => take the average
            val_los_fw, val_score_fw = self._val_epoch(val_loader_fw)
            val_los_rc, val_score_rc = self._val_epoch(val_loader_rc)

            val_los = (val_los_fw + val_los_rc)/2
            val_score = (val_score_fw + val_score_rc)/2

            # print(f'Finished validating epoch [{epoch}/{self.num_epochs}], mean loss: {train_los}, mean score: {train_score}')

            # save checkpoint
            torch.save(self.model.state_dict(), f'{output}/epoch_{epoch}.th')
            if val_los < self.early_stoper.min_validation_loss:
                print('best model so far... saving model....')
                is_improve = True
                torch.save(self.model.state_dict(), f'{output}/best_model.th')

            self.train_stat.append([epoch, train_los, train_score, is_improve])
            self.val_stat.append([epoch, val_los, val_score, is_improve])
            # write out statistics
            self._list_to_csv(f'{output}/train_log.csv', self.train_stat)
            self._list_to_csv(f'{output}/val_log.csv', self.val_stat)

            if self.early_stoper.early_stop(val_los):
                break





