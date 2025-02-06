import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import utils
from config import TrainConfig
from torch.optim.lr_scheduler import ReduceLROnPlateau

config = TrainConfig()

device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
config.print_params(logger.info)

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last validation improvement.
                            Default: 5
            verbose (bool): If True, prints a message for each improvement.
                            Default: False
            delta (float): Minimum change in monitored value to consider as improvement.
                           Default: 0
            path (str): Path for saving the best model checkpoint.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.best_accuracy = 0

    def __call__(self, val_accuracy, model):
        score = val_accuracy
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_accuracy, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_accuracy, model)
            self.counter = 0

    def save_checkpoint(self, val_accuracy, model):
        '''Saves model when validation accuracy improves.'''
        if self.verbose:
            print(f'Validation accuracy improved ({self.best_accuracy:.6f} --> {val_accuracy:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.best_accuracy = val_accuracy

def main():
    logger.info("Logger is set - training start")

    # set default gpu device id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = True

    best_acc = 0.0
    best_wa = 0.0
    best_f1s = 0.0  # Initialize best F1 score
    best_train_wa = 0.0
    train_accuracies = []  # List to store training accuracies
    val_accuracies = []    # List to store validation accuracies

   

    # Initialize the scheduler
   
    # Iterate over the folds for cross-validation
    for fold in range(1,6):
        train_accuracies_fold = []  # List to store training accuracies
        val_accuracies_fold = []
        weighted_accuracies_train_fold = []
        weighted_accuracies_test_fold = []
        logger.info(f"Starting Fold {fold}/{config.num_folds}")
        
        # Load data for the current fold
        input_size, input_channels, n_classes, train_data, valid_data = utils.get_data(
            config.dataset, config.data_path, cutout_length=0, validation=True, features=config.features, fold=fold)

        model = get_model(config.model, config.features).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        #scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

        # Create data loaders for training and validation
        train_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=config.batch_size,
                                                   shuffle=False,
                                                   num_workers=config.workers,
                                                   pin_memory=True, drop_last=True)

        valid_loader = torch.utils.data.DataLoader(valid_data,
                                                   batch_size=config.batch_size,
                                                   shuffle=False,
                                                   num_workers=config.workers,
                                                   pin_memory=True, drop_last=True)

        # Get class_ids_for_name from valid_data
        class_ids_for_name = valid_data.class_ids_for_name
        class_ids_for_name_tr = train_data.class_ids_for_name
        #early_stopping = EarlyStopping(patience=5, verbose=True, path='best_model.pt')
        

        for epoch in range(config.epochs):
            # Training loop
            train_acc, cnn1_features, cnn2_features, cnn3_features, flattened_features, lstm_features, fc1_features, dropout_features, fc2_features, all_labels, all_outputs, final_output, weighted_train = train(model, optimizer, train_loader, epoch, train_data.get_class_weights(),class_ids_for_name_tr)
            
            
            
            cur_step = (epoch + 1) * len(train_loader)

            # Validation loop
            acc, wa, f1s, cnn1_out, cnn2_out, cnn3_out, flattened, lstm_out, fc1_out, dropout_out, fc2_out, logits_output, labels, output = validate(valid_loader, model, epoch, cur_step, valid_data.get_class_weights(), class_ids_for_name)

            train_accuracies.append(train_acc)
            val_accuracies.append(acc)
            
            train_accuracies_fold.append(train_acc)
            val_accuracies_fold.append(acc)
            weighted_accuracies_train_fold.append(weighted_train)
            weighted_accuracies_test_fold.append(wa)
            #scheduler.step(acc)

            #current_lr = optimizer.param_groups[0]['lr']
            #logger.info(f"Epoch {epoch+1}: Current Learning Rate: {current_lr}")
            
            #early_stopping(acc, model)
       

            # Save model if it is the best based on accuracy or F1 score
            is_best = False
            if best_acc < acc:
                best_acc = acc
                is_best = True

            if best_wa < wa:
                best_wa = wa
                
            if best_train_wa < weighted_train:
                best_train_wa = weighted_train

            if best_f1s < f1s:  # Compare F1 score
                best_f1s = f1s
                is_best = True

            utils.save_checkpoint(model, config.path, is_best)

            # Save features and results for the current fold
            save_fold_data(epoch, cnn1_features, cnn2_features, cnn3_features, flattened_features, lstm_features, fc1_features, dropout_features, fc2_features, final_output, all_labels, all_outputs, cnn1_out, cnn2_out, cnn3_out, flattened, lstm_out, fc1_out, dropout_out, fc2_out, logits_output, labels, output, fold)
            
            '''if early_stopping.early_stop:
                print("Early stopping triggered!")
                break'''

        logger.info(f"Fold {fold}/{config.num_folds} - Best Accuracy: {best_acc:.4%}, Best F1 Score: {best_f1s:.4f}, Best wa train Score: {best_train_wa:.4f},Best wa val Score: {best_wa:.4f} ")

        fold_dir = os.path.join(config.plot_path, f'fold_{fold}_epoch_{epoch}')
        fold_dir_tr = os.path.join(config.plot_path, f'overall_fold_{fold}_epoch_{epoch}')
        os.makedirs(fold_dir, exist_ok=True)
        os.makedirs(fold_dir_tr, exist_ok=True)
        plot_accuracies(weighted_accuracies_train_fold, weighted_accuracies_test_fold, fold_dir)
        plot_accuracies(train_accuracies_fold, val_accuracies_fold, fold_dir_tr)
    # After all folds, print the final best results
    logger.info("Final best Accuracy = {:.4%}".format(best_acc))
    logger.info("Final best Weighted Accuracy Val = {:.4%}".format(best_wa))
    logger.info("Final best Weighted Accuracy Train = {:.4%}".format(best_train_wa))
    logger.info("Final best F1 Score = {:.4f}".format(best_f1s))

    plot_accuracies(train_accuracies, val_accuracies, config.plot_path)
    

def save_fold_data(epoch, cnn1_features, cnn2_features, cnn3_features, flattened_features, lstm_features, fc1_features, dropout_features, fc2_features, final_output, all_labels, all_outputs, cnn1_out, cnn2_out, cnn3_out, flattened, lstm_out, fc1_out, dropout_out, fc2_out, logits_output, labels, output, fold):
    """ Save the features and outputs for each fold """
    fold_dir = os.path.join(config.features_path, f'fold_{fold}_epoch_{epoch}')
    os.makedirs(fold_dir, exist_ok=True)
    
    # Save train and validation features
    torch.save(cnn1_features, os.path.join(fold_dir, 'cnn1_features.pt'))
    torch.save(cnn2_features, os.path.join(fold_dir, 'cnn2_features.pt'))
    torch.save(cnn3_features, os.path.join(fold_dir, 'cnn3_features.pt'))
    torch.save(flattened_features, os.path.join(fold_dir, 'flattened_features.pt'))
    torch.save(lstm_features, os.path.join(fold_dir, 'lstm_features.pt'))
    torch.save(fc1_features, os.path.join(fold_dir, 'fc1_features.pt'))
    torch.save(dropout_features, os.path.join(fold_dir, 'dropout_features.pt'))
    torch.save(fc2_features, os.path.join(fold_dir, 'fc2_features.pt'))
    torch.save(final_output, os.path.join(fold_dir, 'final_output.pt'))
    np.save(os.path.join(fold_dir, "train_labels.npy"), all_labels)
    np.save(os.path.join(fold_dir, "train_outputs.npy"), all_outputs)
    torch.save(cnn1_out, os.path.join(fold_dir, 'cnn1_out.pt'))
    torch.save(cnn2_out, os.path.join(fold_dir, 'cnn2_out.pt'))
    torch.save(cnn3_out, os.path.join(fold_dir, 'cnn3_out.pt'))
    torch.save(flattened, os.path.join(fold_dir, 'flattened.pt'))
    torch.save(lstm_out, os.path.join(fold_dir, 'lstm_out.pt'))
    torch.save(fc1_out, os.path.join(fold_dir, 'fc1_out.pt'))
    torch.save(dropout_out, os.path.join(fold_dir, 'dropout_out.pt'))
    torch.save(fc2_out, os.path.join(fold_dir, 'fc2_out.pt'))
    torch.save(logits_output, os.path.join(fold_dir, 'logits_output.pt'))
    np.save(os.path.join(fold_dir, "val_labels.npy"), labels)
    np.save(os.path.join(fold_dir, "val_outputs.npy"), output)

'''def main():
    logger.info("Logger is set - training start")

    # set default gpu device id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True

    input_size, input_channels, n_classes, train_data, valid_data = utils.get_data(
        config.dataset, config.data_path, cutout_length=0, validation=True, features=config.features, fold=config.fold)

    model = get_model(config.model, config.features).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    n_train = len(train_data)
    split = n_train // 5
    indices = list(range(n_train))

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               shuffle=False,
                                               num_workers=config.workers,
                                               pin_memory=True, drop_last=True)

    valid_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=config.batch_size,
                                               shuffle=False,
                                               num_workers=config.workers,
                                               pin_memory=True, drop_last=True)
    best_acc = 0.
    best_wa = 0.
    best_f1s
    for epoch in range(config.epochs):
        train(model, optimizer, train_loader, epoch)
        cur_step = (epoch + 1) * len(train_loader)

        acc, wa = validate(valid_loader, model, epoch, cur_step, valid_data.get_class_weights())

        if best_acc < acc:
            best_acc = acc
            is_best = True
        else:
            is_best = False

        if best_wa < wa:
            best_wa = wa

        utils.save_checkpoint(model, config.path, is_best)

    logger.info("Final best Accuracy = {:.4%}".format(best_acc))'''


'''def main():
    logger.info("Logger is set - training start")

    # set default gpu device id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True

    input_size, input_channels, n_classes, train_data, valid_data = utils.get_data(
        config.dataset, config.data_path, cutout_length=0, validation=True, features=config.features, fold=config.fold)

    model = get_model(config.model, config.features).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               shuffle=False,
                                               num_workers=config.workers,
                                               pin_memory=True, drop_last=True)

    valid_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=config.batch_size,
                                               shuffle=False,
                                               num_workers=config.workers,
                                               pin_memory=True, drop_last=True)

    best_acc = 0.0
    best_wa = 0.0
    best_f1s = 0.0  # Initialize best F1 score

    for epoch in range(config.epochs):
        train(model, optimizer, train_loader, epoch)
        cur_step = (epoch + 1) * len(train_loader)

        acc, wa, f1s = validate(valid_loader, model, epoch, cur_step, valid_data.get_class_weights())

        is_best = False
        if best_acc < acc:
            best_acc = acc
            is_best = True  # Update flag to indicate this is the best model based on accuracy

        if best_wa < wa:
            best_wa = wa

        if best_f1s < f1s:  # Compare F1 score
            best_f1s = f1s
            is_best = True  # Update flag to save based on F1 score as well

        utils.save_checkpoint(model, config.path, is_best)

    logger.info("Final best Accuracy = {:.4%}".format(best_acc))
    logger.info("Final best Weighted Accuracy = {:.4%}".format(best_wa))
    logger.info("Final best F1 Score = {:.4f}".format(best_f1s))'''

'''def main():
    logger.info("Logger is set - training start")

    # set default gpu device id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True

    input_size, input_channels, n_classes, train_data, valid_data = utils.get_data(
        config.dataset, config.data_path, cutout_length=0, validation=True, features=config.features, fold=config.fold)

    model = get_model(config.model, config.features).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               shuffle=False,
                                               num_workers=config.workers,
                                               pin_memory=True, drop_last=True)

    valid_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=config.batch_size,
                                               shuffle=False,
                                               num_workers=config.workers,
                                               pin_memory=True, drop_last=True)

    best_acc = 0.0
    best_wa = 0.0
    best_f1s = 0.0  # Initialize best F1 score
    train_accuracies = []  # List to store training accuracies
    val_accuracies = []    # List to store validation accuracies

    # Extract class_ids_for_name from valid_data
    class_ids_for_name = valid_data.class_ids_for_name
    #early_stopping = EarlyStopping(patience=5, verbose=True, path='best_model.pt')
    for epoch in range(config.epochs):
        train_acc, cnn1_features, cnn2_features, cnn3_features, flattened_features, lstm_features, fc1_features, dropout_features, fc2_features, all_labels, all_outputs, final_output = train(model, optimizer, train_loader, epoch)
        
        #train_acc, cnn1_features, cnn2_features, flattened_features, lstm_features, fc1_features, dropout_features, fc2_features, all_labels, all_outputs, final_output = train(model, optimizer, train_loader, epoch)
        
        cur_step = (epoch + 1) * len(train_loader)

        acc, wa, f1s,cnn1_out, cnn2_out, cnn3_out, flattened, lstm_out, fc1_out, dropout_out, fc2_out, logits_output, labels, output = validate(valid_loader, model, epoch, cur_step, valid_data.get_class_weights(), class_ids_for_name)

        train_accuracies.append(train_acc)
        val_accuracies.append(acc)
        
        #early_stopping(acc, model)
        
        #if early_stopping.early_stop:
            #print("Early stopping triggered!")
            #break
        
        is_best = False
        if best_acc < acc:
            best_acc = acc
            is_best = True  # Update flag to indicate this is the best model based on accuracy

        if best_wa < wa:
            best_wa = wa

        if best_f1s < f1s:  # Compare F1 score
            best_f1s = f1s
            is_best = True  # Update flag to save based on F1 score as well

        utils.save_checkpoint(model, config.path, is_best)
        
        epoch_dir = os.path.join(config.features_path, f'epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)
        
        val_dir = os.path.join(config.val_features_path, f'epoch_{epoch}')
        os.makedirs(val_dir, exist_ok=True)
        
        torch.save(cnn1_features, os.path.join(epoch_dir,'cnn1_features.pt'))
        torch.save(cnn2_features, os.path.join(epoch_dir,'cnn2_features.pt'))
        torch.save(cnn3_features, os.path.join(epoch_dir,'cnn3_features.pt'))
        torch.save(flattened_features, os.path.join(epoch_dir,'flattened_features.pt'))
        torch.save(lstm_features, os.path.join(epoch_dir,'lstm_features.pt'))
        torch.save(fc1_features, os.path.join(epoch_dir,'fc1_features.pt'))
        torch.save(dropout_features, os.path.join(epoch_dir,'dropout_features.pt'))
        torch.save(fc2_features, os.path.join(epoch_dir,'fc2_features.pt'))
        torch.save(final_output, os.path.join(epoch_dir,'final_output.pt'))
        
        np.save(os.path.join(config.features_path, f"train_labels_epoch_{epoch}.npy"), all_labels)
        np.save(os.path.join(config.features_path, f"train_outputs_epoch_{epoch}.npy"), all_outputs)
        
        torch.save(cnn1_out, os.path.join(val_dir,'cnn1_features.pt'))
        torch.save(cnn2_out, os.path.join(val_dir,'cnn2_features.pt'))
        torch.save(cnn3_out, os.path.join(val_dir,'cnn3_features.pt'))
        torch.save(flattened, os.path.join(val_dir,'flattened_features.pt'))
        torch.save(lstm_out, os.path.join(val_dir,'lstm_features.pt'))
        torch.save(fc1_out, os.path.join(val_dir,'fc1_features.pt'))
        torch.save(dropout_out, os.path.join(val_dir,'dropout_features.pt'))
        torch.save(fc2_out, os.path.join(val_dir,'fc2_features.pt'))
        torch.save(logits_output, os.path.join(val_dir,'final_output.pt'))
        
        np.save(os.path.join(config.val_features_path, f"val_labels_epoch_{epoch}.npy"), labels)
        np.save(os.path.join(config.val_features_path, f"val_outputs_epoch_{epoch}.npy"), output)

    logger.info("Final best Accuracy = {:.4%}".format(best_acc))
    logger.info("Final best Weighted Accuracy = {:.4%}".format(best_wa))
    logger.info("Final best F1 Score = {:.4f}".format(best_f1s))

    plot_accuracies(train_accuracies, val_accuracies, config.plot_path)'''


def train(model, optimizer, train_loader, epoch, class_weights, class_ids_for_name):
    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()
    weighted_accuracy = utils.AverageMeter()
    weighted_accuracy2 = utils.AverageMeter()

    all_labels = []  # List to store all labels
    all_outputs = []  # List to store model outputs
    cur_step = epoch * len(train_loader)
    #weighted_accura = []

    for step, (trn_x, trn_y) in enumerate(train_loader):
        trn_x = trn_x.to(device)
        trn_y = torch.nn.functional.one_hot(trn_y, num_classes=4).float().to(device)
        N = trn_x.size(0)

        model.train()
        optimizer.zero_grad()

        cnn1_features, cnn2_features, cnn3_features, flattened_features, lstm_features, fc1_features, dropout_features, fc2_features, output= model(trn_x)
        #cnn1_features, cnn2_features, flattened_features, lstm_features, fc1_features, dropout_features, fc2_features, output= model(trn_x)
        loss = torch.nn.functional.mse_loss(output, trn_y)
        #loss = torch.nn.functional.cross_entropy(output, trn_y)
        loss.backward(retain_graph=True)

        optimizer.step()
        acc = utils.accuracy(output, trn_y)
        wa, f1_per_class, recall_per_class, precision_per_class = utils.scores(output, trn_y, class_ids_for_name, class_weights)
        weighted_acc =  utils.weighted_accuracy(output, trn_y, class_weights)
        weighted_acc2,_,_,_ = utils.scores_old(output, trn_y, class_weights)

        losses.update(loss.item(), N)
        accuracy.update(acc.item(), N)
        weighted_accuracy.update(wa, N)
        weighted_accuracy2.update(weighted_acc2, N)
        
        all_labels.append(trn_y.cpu().numpy())  # Converting labels to numpy and storing
        all_outputs.append(output.cpu().detach().numpy()) 
       

        if step % config.print_freq == 0 or step == len(train_loader) - 1:
            logger.info("Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} Acc {accuracy.avg:.2%} Weighted Acc {weighted_accuracy.avg:.2%} Weighted Acc old {weighted_accuracy2.avg:.2%}".format(
                epoch + 1, config.epochs, step, len(train_loader) - 1, losses=losses, accuracy=accuracy, weighted_accuracy = weighted_accuracy, weighted_accuracy2=weighted_accuracy2
            ))

        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/acc', acc.item(), cur_step)
        cur_step += 1
        
    all_labels_np = np.concatenate(all_labels, axis=0)  # Combine all batches into a single array
    all_outputs_np = np.concatenate(all_outputs, axis=0)

    logger.info("Train: [{:2d}/{}] Final Acc {:.4%}".format(epoch + 1, config.epochs, accuracy.avg))
    logger.info("Train: [{:2d}/{}] Final Weighted Acc {:.4%}".format(epoch + 1, config.epochs, weighted_accuracy.avg))
    
    return accuracy.avg, cnn1_features, cnn2_features, cnn3_features, flattened_features, lstm_features, fc1_features, dropout_features, fc2_features, all_labels_np, all_outputs_np,  output,weighted_accuracy.avg
    #return accuracy.avg, cnn1_features, cnn2_features, flattened_features, lstm_features, fc1_features, dropout_features, fc2_features, all_labels_np, all_outputs_np,  output'''


''' def validate(valid_loader, model, epoch, cur_step, class_weights):
    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()
    weighted_accuracy = utils.AverageMeter()
    f1s = utils.AverageMeter()
    recalls = utils.AverageMeter()
    precesions = utils.AverageMeter()

    model.eval()

    predictions = []
    true_labels = []

    with (torch.no_grad()):
        for step, (X, y) in enumerate(valid_loader):
            X = X.to(device, non_blocking=True)
            y = torch.nn.functional.one_hot(y, num_classes=4).float().to(device)
            N = X.size(0)

            logits = model(X)
            loss = torch.nn.functional.mse_loss(logits, y)
            losses.update(loss.item(), N)

            acc = utils.accuracy(logits, y)
            accuracy.update(acc.item(), N)

            wa, f1, r, p = utils.scores(logits, y, class_weights)

            weighted_accuracy.update(wa, N)
            f1s.update(f1, N)
            recalls.update(r, N)
            precesions.update(p, N)

            if step % config.print_freq == 0 or step == len(valid_loader) - 1:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Acc {accuracy.avg:.2%}".format(
                        epoch + 1, config.epochs, step, len(valid_loader) - 1, losses=losses,
                        accuracy=accuracy))

    writer.add_scalar('val/loss', losses.avg, cur_step)
    writer.add_scalar('val/acc', accuracy.avg, cur_step)
    writer.add_scalar('val/wa', weighted_accuracy.avg, cur_step)

    logger.info(
        "Valid: [{:2d}/{}] Final Acc {:.4%}".format(epoch + 1, config.epochs, accuracy.avg))

    return accuracy.avg, weighted_accuracy.avg'''

'''def validate(valid_loader, model, epoch, cur_step, class_weights):
    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()
    weighted_accuracy = utils.AverageMeter()
    
    # List of meters for each class (assuming 4 classes)
    f1s_per_class = [utils.AverageMeter() for _ in range(4)]
    recalls_per_class = [utils.AverageMeter() for _ in range(4)]
    precisions_per_class = [utils.AverageMeter() for _ in range(4)]

    model.eval()

    predictions = []
    true_labels = []

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X = X.to(device, non_blocking=True)
            y_one_hot = torch.nn.functional.one_hot(y, num_classes=4).float().to(device)
            N = X.size(0)

            logits = model(X)
            loss = torch.nn.functional.mse_loss(logits, y_one_hot)
            losses.update(loss.item(), N)

            acc = utils.accuracy(logits, y_one_hot)
            accuracy.update(acc.item(), N)

            # Scores now return metrics per class
            wa, f1_per_class, recall_per_class, precision_per_class = utils.scores(logits, y, class_weights)

            weighted_accuracy.update(wa, N)
            
            # Update AverageMeter for each class's F1, recall, and precision
            for i in range(4):
                f1s_per_class[i].update(f1_per_class[i], N)
                recalls_per_class[i].update(recall_per_class[i], N)
                precisions_per_class[i].update(precision_per_class[i], N)

            if step % config.print_freq == 0 or step == len(valid_loader) - 1:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Acc {accuracy.avg:.2%}".format(
                        epoch + 1, config.epochs, step, len(valid_loader) - 1, losses=losses,
                        accuracy=accuracy))
                
                # Log F1, Recall, Precision per class
                for i in range(4):
                    logger.info(f"Class {i} - F1: {f1s_per_class[i].avg:.3f}, "
                                f"Recall: {recalls_per_class[i].avg:.3f}, "
                                f"Precision: {precisions_per_class[i].avg:.3f}")

    # Write aggregated metrics to TensorBoard or log them
    writer.add_scalar('val/loss', losses.avg, cur_step)
    writer.add_scalar('val/acc', accuracy.avg, cur_step)
    writer.add_scalar('val/wa', weighted_accuracy.avg, cur_step)

    # Log per class F1 scores in TensorBoard
    for i in range(4):
        writer.add_scalar(f'val/f1_class_{i}', f1s_per_class[i].avg, cur_step)
        writer.add_scalar(f'val/recall_class_{i}', recalls_per_class[i].avg, cur_step)
        writer.add_scalar(f'val/precision_class_{i}', precisions_per_class[i].avg, cur_step)

    logger.info(
        "Valid: [{:2d}/{}] Final Acc {:.4%}".format(epoch + 1, config.epochs, accuracy.avg))

    # Return overall average accuracy, weighted accuracy, and average F1 across all classes
    avg_f1 = sum([f.avg for f in f1s_per_class]) / 4
    return accuracy.avg, weighted_accuracy.avg, avg_f1'''


def validate(valid_loader, model, epoch, cur_step, class_weights, class_ids_for_name):
    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()
    weighted_accuracy = utils.AverageMeter()
    f1s = utils.AverageMeter()
    recalls = utils.AverageMeter()
    precisions = utils.AverageMeter()

    model.eval()

    predictions = []
    true_labels = []

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X = X.to(device, non_blocking=True)
            y = torch.nn.functional.one_hot(y, num_classes=4).float().to(device)
            N = X.size(0)

            #_,_,_,_,_,_,_,_,logits = model(X)
            cnn1_out, cnn2_out, cnn3_out, flattened, lstm_out, fc1_out, dropout_out, fc2_out, logits = model(X)
            #_,_,_,_,_,_,_, logits = model(X)
            loss = torch.nn.functional.mse_loss(logits, y)
            #loss = torch.nn.functional.cross_entropy(logits, y)
            losses.update(loss.item(), N)

            acc = utils.accuracy(logits, y)
            accuracy.update(acc.item(), N)

            wa, f1_per_class, recall_per_class, precision_per_class = utils.scores(logits, y, class_ids_for_name, class_weights)
            weighted_acc =  utils.weighted_accuracy(logits, y, class_weights)
            weighted_acc2 =  utils.scores_old(logits, y, class_weights)

            weighted_accuracy.update(wa, N) 
            f1s.update(np.mean(f1_per_class), N)  # Average F1 score across classes
            recalls.update(np.mean(recall_per_class), N)
            precisions.update(np.mean(precision_per_class), N)

            if step % config.print_freq == 0 or step == len(valid_loader) - 1:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Acc {accuracy.avg:.2%} Weighted Acc {weighted_accuracy.avg:.2%}".format(
                        epoch + 1, config.epochs, step, len(valid_loader) - 1, losses=losses,
                        accuracy=accuracy, weighted_accuracy = weighted_accuracy))

            true_labels.append(y.cpu().numpy())  # Converting labels to numpy and storing
            predictions.append(logits.cpu().detach().numpy()) 
     
    all_labels_np = np.concatenate(true_labels, axis=0)  # Combine all batches into a single array
    all_outputs_np = np.concatenate(predictions, axis=0)
    # Log class-wise F1 scores
    class_names = list(class_ids_for_name.keys())
    for i, class_name in enumerate(class_names):
        logger.info(f"Class '{class_name}' - F1 Score: {f1_per_class[i]:.3f}")

    writer.add_scalar('val/loss', losses.avg, cur_step)
    writer.add_scalar('val/acc', accuracy.avg, cur_step)
    writer.add_scalar('val/wa', weighted_accuracy.avg, cur_step)
    writer.add_scalar('val/f1s', f1s.avg, cur_step)

    logger.info(
        "Valid: [{:2d}/{}] Final Acc {:.4%} Final Weighted Acc {:.4%}".format(epoch + 1, config.epochs, accuracy.avg, weighted_accuracy.avg))

    return accuracy.avg, weighted_accuracy.avg, f1s.avg, cnn1_out, cnn2_out, cnn3_out, flattened, lstm_out, fc1_out, dropout_out, fc2_out, logits, all_labels_np, all_outputs_np


def plot_accuracies(train_accuracies, val_accuracies, save_path):
    """Function to plot and save training and validation accuracies."""
    epochs = range(1, len(train_accuracies) + 1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracies, 'b', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracies')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(os.path.join(save_path,"accuracy.png"))  # Save plot to file
    print(f"Plot saved to {save_path}")
    
    #plt.show()  # Optionally display the plot after saving
    
def get_model(model_type, features):
    model_type = model_type.lower()
    features = features.lower()

    if features == 'iemocapmfcctests':
        from mfcc_model import MFCC_CNNModel, MFCC_RNNModel, MFCC_LSTMModel, MFCC_CNNLSTMModel, MFCC_CNNLSTMAttModel
        if model_type == 'cnn':
            return MFCC_CNNModel()
        elif model_type == 'rnn':
            return MFCC_RNNModel()
        elif model_type == 'lstm':
            return MFCC_LSTMModel()
        elif model_type == 'cnnlstm':
            return MFCC_CNNLSTMModel()
        elif model_type == 'cnnlstmatt':
            return MFCC_CNNLSTMAttModel()
        else:
            raise Exception('invalid model type')
    else:
        raise Exception('invalid feature')


if __name__ == "__main__":
    main()
