import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset
import os
import random
import numpy as np
import copy  # [Added] Import copy module to save best model weights

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# >>>>>>>>>> Fix all random seeds <<<<<<<<<<
SEED = 44
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # if multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# >>>>>>>>>> End of seed setting <<<<<<<<<<


# [Added] Add patience=15 to parameters
def train(model, lr, num_epochs, X_train, Y_train, X_val, Y_val, edge_index, rt_meas_dim=76, device='cuda', patience=6):
    # [Fix 1] Ensure all input data is on the specified device
    # X_train = X_train.to(device)
    # Y_train = Y_train.to(device)
    # X_val = X_val.to(device)
    # Y_val = Y_val.to(device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    edge_index = edge_index.to(device)

    # Ensure model is also on device (double insurance)
    model.to(device)

    # weighted cross entropy loss
    num_class_0 = (Y_train == 0).sum().item()
    num_class_1 = (Y_train == 1).sum().item()

    # Prevent division by zero
    if num_class_1 == 0:
        pos_weight = torch.tensor([1.0], dtype=torch.float32, device=device)
    else:
        pos_weight = torch.tensor([num_class_0 / num_class_1], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean').to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    action_mask = model.action_mask

    if model.name == 'NN' or model.name == 'DCNN':
        X_train = X_train[:, action_mask, -rt_meas_dim:].clone()
        X_val = X_val[:, action_mask, -rt_meas_dim:].clone()

    dataset = TensorDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    stat = {}
    stat['loss_train'] = []
    stat['loss_val'] = []
    stat['acc_train'] = []
    stat['acc_val'] = []

    # ==================== [Added] Early stopping initialization variables ====================
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())  # Record initial weights
    # ================================================================

    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            if model.name == 'NN':
                output = model(batch_X)
                loss = criterion(output, batch_y)

            elif model.name == 'GAT':
                loss = 0
                for i in range(len(batch_X)):
                    output = model(batch_X[i], edge_index)
                    loss += criterion(output[action_mask], batch_y[i])
                loss /= len(batch_X)

            elif model.name == 'transformer':
                loss = 0
                for i in range(len(batch_X)):
                    output = model(batch_X[i])
                    if output.dim() == 2 and output.size(0) == 1:
                        output = output.squeeze(0)
                    selected_output = output[action_mask]
                    target = batch_y[i]
                    if target.dim() > 1 and target.size(-1) == 1:
                        target = target.squeeze(-1)
                    if selected_output.shape != target.shape:
                        target = target.view_as(selected_output)
                    loss += criterion(selected_output, target)
                loss /= len(batch_X)

            elif model.name in ['mainModel', 'WithoutGCN', 'WithoutTransformer', 'WithoutCNN']:
                loss = 0
                for i in range(len(batch_X)):
                    output = model(batch_X[i], edge_index)
                    if output.dim() == 2 and output.size(0) == 1:
                        output = output.squeeze(0)
                    selected_output = output[action_mask]
                    target = batch_y[i]
                    if target.dim() > 1 and target.size(-1) == 1:
                        target = target.squeeze(-1)
                    if selected_output.shape != target.shape:
                        target = target.view_as(selected_output)
                    loss += criterion(selected_output, target)
                loss /= len(batch_X)

            elif model.name == 'DCNN':
                output = model(batch_X).squeeze(-1)
                loss = criterion(output, batch_y)

            else:
                output = model(batch_X, edge_index)
                loss = criterion(output[:, action_mask], batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluation
        loss_train, acc_train = evaluate_loss_acc(model, X_train, Y_train, criterion, edge_index, device)
        loss_val, acc_val = evaluate_loss_acc(model, X_val, Y_val, criterion, edge_index, device=device)

        stat['loss_train'].append(loss_train)
        stat['loss_val'].append(loss_val)
        stat['acc_train'].append(acc_train)
        stat['acc_val'].append(acc_val)

        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            print('Epoch: {:03d}, Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}'.format(
                epoch + 1, loss_train, acc_train, loss_val, acc_val))

        # ==================== [Added] Core logic for early stopping ====================
        # If current validation loss is lower than the historical minimum loss, the model has improved
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            epochs_no_improve = 0  # Reset patience counter
            best_model_wts = copy.deepcopy(model.state_dict())  # Deep copy and update best model weights
        else:
            # If no improvement, increment patience counter
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'\n[Early Stopping Triggered] Validation Loss has not decreased for {patience} consecutive epochs.')
                print(f'Stopping at epoch {epoch + 1}, restoring best model weights (Val Loss: {best_val_loss:.4f})')
                break  # Exit loop and end training
        # ================================================================

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
        print(f"[{model.__class__.__name__}] Training Peak Memory: {peak_mem:.2f} MB")

    # ==================== [Added] Load best weights at the end ====================
    # Whether training ends due to early stopping or completes all epochs, reload the best performing weights to the model
    model.load_state_dict(best_model_wts)
    # ================================================================

    model.stat = stat


def evaluate_loss_acc(model, X, y, criterion, edge_index, device='cuda'):
    model.eval()
    mask = model.action_mask
    loss, acc = 0, 0

    # [Fix 3] Ensure evaluation data is also on device
    X = X.to(device)
    y = y.to(device)
    edge_index = edge_index.to(device)

    with torch.no_grad():
        if model.name == 'NN' or model.name == 'DCNN':
            output = model(X).squeeze(-1)
            loss = criterion(output, y)
            y_pred = torch.sigmoid(output) > 0.5
            acc = (y_pred == y).sum().item() / (y.shape[0] * y.shape[1])

        elif model.name == 'GAT':
            true_pred = []
            # Evaluate GAT sample by sample
            for i in range(len(X)):
                output = model(X[i], edge_index)
                loss += criterion(output[mask], y[i])
                y_pred = torch.sigmoid(output[mask]) > 0.5
                true_pred.append(y_pred)
            loss /= len(X)
            acc = (torch.stack(true_pred, dim=0) == y).sum().item() / (y.shape[0] * y.shape[1])
        elif model.name == 'transformer':
            total_loss = 0
            true_pred = []

            for i in range(len(X)):
                output = model(X[i])  # (1, N)
                if output.dim() == 2 and output.size(0) == 1:
                    output = output.squeeze(0)  # (N,)

                selected_out = output[mask]
                target = y[i]
                if target.dim() > 1: target = target.squeeze(-1)
                target = target.view_as(selected_out)

                total_loss += criterion(selected_out, target)

                y_pred = torch.sigmoid(selected_out) > 0.5
                true_pred.append(y_pred)

            loss = total_loss / len(X)
            stacked_pred = torch.stack(true_pred, dim=0)  # (Batch, Num_Mask_Nodes)

            # Calculate accuracy
            # Shape of y should be (Batch, Num_Mask_Nodes)
            target_y = y
            if target_y.dim() > 2: target_y = target_y.squeeze(-1)

            acc = (stacked_pred == target_y).sum().item() / target_y.numel()
        elif model.name in ['mainModel', 'WithoutGCN', 'WithoutTransformer', 'WithoutCNN']:
            # print("============================================================Entered here")
            total_loss = 0
            true_pred = []

            for i in range(len(X)):
                output = model(X[i],edge_index)  # (1, N)
                if output.dim() == 2 and output.size(0) == 1:
                    output = output.squeeze(0)  # (N,)

                selected_out = output[mask]
                target = y[i]
                if target.dim() > 1: target = target.squeeze(-1)
                target = target.view_as(selected_out)

                total_loss += criterion(selected_out, target)

                y_pred = torch.sigmoid(selected_out) > 0.5
                true_pred.append(y_pred)

            loss = total_loss / len(X)
            stacked_pred = torch.stack(true_pred, dim=0)  # (Batch, Num_Mask_Nodes)

            # Calculate accuracy
            # Shape of y should be (Batch, Num_Mask_Nodes)
            target_y = y
            if target_y.dim() > 2: target_y = target_y.squeeze(-1)

            acc = (stacked_pred == target_y).sum().item() / target_y.numel()

        else:
            output = model(X, edge_index)
            loss = criterion(output[:, mask], y)
            y_pred = torch.sigmoid(output[:, mask]) > 0.5
            acc = (y_pred == y).sum().item() / (y.shape[0] * y.shape[1])

    return loss.item(), acc  # loss may be tensor, convert to item


def predict_prob(model, X, edge_index, rt_meas_dim=76, device='cuda'):
    model = model.to(device)
    model.eval()
    mask = model.action_mask

    # [Fix 4] Ensure input is on device
    X = X.to(device)
    edge_index = edge_index.to(device)

    # Initialize prob on device
    prob = torch.zeros((len(X), len(mask), 2), dtype=torch.float32, device=device)

    with torch.no_grad():
        if model.name == 'NN' or model.name == 'DCNN':
            # Slice and ensure on device
            input_x = X[:, mask, -rt_meas_dim:].to(device)
            output = model(input_x).squeeze(-1)  # Add .squeeze(-1)
            prob_1 = torch.sigmoid(output)
            prob = torch.stack([1 - prob_1, prob_1], dim=-1)  # dim=-1 for 2D compatibility

        elif model.name == 'GAT':
            prob_1_list = []
            for i in range(len(X)):
                out = model(X[i], edge_index)
                prob_1_list.append(torch.sigmoid(out)[mask])
            prob_1 = torch.stack(prob_1_list, dim=0)
            prob = torch.stack([1 - prob_1, prob_1], dim=2)
        elif model.name == 'transformer':
            prob_1_list = []
            for i in range(len(X)):
                out = model(X[i])  # (1, N)
                if out.dim() == 2 and out.size(0) == 1:
                    out = out.squeeze(0)  # (N,)

                # Filter and sigmoid
                selected_out = out[mask]
                prob_1_list.append(torch.sigmoid(selected_out))

            prob_1 = torch.stack(prob_1_list, dim=0)  # (Batch, Num_Mask_Nodes)
            prob = torch.stack([1 - prob_1, prob_1], dim=2)  # (Batch, Num_Mask_Nodes, 2)
        elif model.name in ['mainModel', 'WithoutGCN', 'WithoutTransformer', 'WithoutCNN']:
            # print("============================================================Entered here")
            prob_1_list = []
            for i in range(len(X)):
                out = model(X[i],edge_index)  # (1, N)
                if out.dim() == 2 and out.size(0) == 1:
                    out = out.squeeze(0)  # (N,)

                # Filter and sigmoid
                selected_out = out[mask]
                prob_1_list.append(torch.sigmoid(selected_out))

            prob_1 = torch.stack(prob_1_list, dim=0)  # (Batch, Num_Mask_Nodes)
            prob = torch.stack([1 - prob_1, prob_1], dim=2)  # (Batch, Num_Mask_Nodes, 2)

        else:
            output = model(X, edge_index)
            prob_1 = torch.sigmoid(output)[:, mask]
            prob = torch.stack([1 - prob_1, prob_1], dim=2)

    return prob


# evaluate_acc and evaluate_performance basically don't need major changes,
# because they call predict_prob. However, for safety, .to(device) checks can also be added,
# but as long as predict_prob handles it, it's usually fine here
def evaluate_acc(model, X, y, edge_index, device='cuda'):
    prob = predict_prob(model, X, edge_index, device=device)
    y = y.to(device)  # Ensure y is on device
    pred = torch.argmax(prob, dim=2)
    accuracy = (pred == y).sum().item() / (y.shape[0] * y.shape[1])
    return accuracy


def evaluate_performance(models, X, y, edge_index, device='cuda'):
    metrics = []
    # Ensure global data is on device
    X = X.to(device)
    y = y.to(device)
    edge_index = edge_index.to(device)

    for name, model in models.items():
        model.eval()
        prob = predict_prob(model, X, edge_index, device=device)
        pred_ts = torch.argmax(prob, dim=2)

        # Ensure shape of y matches pred_ts
        # Note: If y is full node labels while pred_ts is masked, need to correspond here
        # According to previous logic, predict_prob returns masked results (len(mask))
        # So y should also take mask part, or already sliced when generating dataset
        # Assume y here is already masked or shape-matched when passed in

        y_target = y  # If y is full graph, may need y[:, mask] here

        accuracy = (pred_ts == y_target).sum().item() / (y_target.shape[0] * y_target.shape[1])

        # Move back to CPU for sklearn calculation (sklearn doesn't support GPU tensor)
        y_flat = y_target.cpu().flatten().numpy()
        pred_flat = pred_ts.cpu().flatten().numpy()
        prob_flat = prob.cpu().view(-1, 2).numpy()

        conf_matrix = confusion_matrix(y_flat, pred_flat)
        precision, recall, f1, _ = precision_recall_fscore_support(y_flat, pred_flat, average='macro', zero_division=0)

        TN, FP, FN, TP = conf_matrix.ravel()
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        FNR = FN / (FN + TP) if (FN + TP) > 0 else 0

        fpr, tpr, thresholds = roc_curve(y_flat, prob_flat[:, 1])
        roc_auc = auc(fpr, tpr)

        d = {'model': name, 'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP,
             'precision': '{:.4f}'.format(precision), 'recall': '{:.4f}'.format(recall), 'f1': '{:.4f}'.format(f1),
             'auc': '{:.4f}'.format(roc_auc), 'fpr': '{:.4f}'.format(FPR), 'fnr': '{:.4f}'.format(FNR),
             'loss_train': '{:.4f}'.format(model.stat['loss_train'][-1]),
             'loss_val': '{:.4f}'.format(model.stat['loss_val'][-1]),
             'acc_train': '{:.4f}'.format(model.stat['acc_train'][-1]),
             'acc_val': '{:.4f}'.format(model.stat['acc_val'][-1]),
             'accuracy': '{:.4f}'.format(accuracy)
             }
        metrics.append(d)

    return metrics