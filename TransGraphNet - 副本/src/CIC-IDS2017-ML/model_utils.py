import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset
import os
import random
import numpy as np
import copy  # 【新增】引入 copy 模块用于保存最佳模型权重

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# >>>>>>>>>> 固定所有随机种子 <<<<<<<<<<
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


# 【新增】在参数中加入 patience=15
def train(model, lr, num_epochs, X_train, Y_train, X_val, Y_val, edge_index, rt_meas_dim=78, device='cuda', patience=6):
    # 【修复 1】确保所有输入数据都在指定的 device 上
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    X_val = X_val.to(device)
    Y_val = Y_val.to(device)
    edge_index = edge_index.to(device)

    # 确保模型也在 device 上 (双重保险)
    model.to(device)

    # weighted cross entropy loss
    num_class_0 = (Y_train == 0).sum().item()
    num_class_1 = (Y_train == 1).sum().item()

    # 防止除以零
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

    # ==================== 【新增】早停初始化变量 ====================
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())  # 记录初始权重
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

        # 评估
        loss_train, acc_train = evaluate_loss_acc(model, X_train, Y_train, criterion, edge_index, device)
        loss_val, acc_val = evaluate_loss_acc(model, X_val, Y_val, criterion, edge_index, device=device)

        stat['loss_train'].append(loss_train)
        stat['loss_val'].append(loss_val)
        stat['acc_train'].append(acc_train)
        stat['acc_val'].append(acc_val)

        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            print('Epoch: {:03d}, Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}'.format(
                epoch + 1, loss_train, acc_train, loss_val, acc_val))

        # ==================== 【新增】早停核心逻辑 ====================
        # 如果当前验证集 loss 比历史最低 loss 还要低，说明模型有进步
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            epochs_no_improve = 0  # 耐心值计数器清零
            best_model_wts = copy.deepcopy(model.state_dict())  # 深拷贝并更新最佳模型权重
        else:
            # 如果没进步，耐心值加 1
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'\\n[Early Stopping Triggered] 连续 {patience} 轮 Validation Loss 未降低。')
                print(f'停止在第 {epoch + 1} 轮, 恢复最佳模型权重 (Val Loss: {best_val_loss:.4f})')
                break  # 跳出循环，结束训练
        # ================================================================

    # ==================== 【新增】结束时加载最佳权重 ====================
    # 无论是因为早停结束还是跑完了全部 epoch，都把表现最好的那一次权重重新加载给模型
    model.load_state_dict(best_model_wts)
    # ================================================================

    model.stat = stat


def evaluate_loss_acc(model, X, y, criterion, edge_index, device='cuda'):
    model.eval()
    mask = model.action_mask
    loss, acc = 0, 0

    # 【修复 3】确保评估数据也在 device 上
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
            # GAT 逐个评估
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

            # 计算准确率
            # y 的形状应该是 (Batch, Num_Mask_Nodes)
            target_y = y
            if target_y.dim() > 2: target_y = target_y.squeeze(-1)

            acc = (stacked_pred == target_y).sum().item() / target_y.numel()
        elif model.name in ['mainModel', 'WithoutGCN', 'WithoutTransformer', 'WithoutCNN']:
            # print("============================================================进入此处了")
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

            # 计算准确率
            # y 的形状应该是 (Batch, Num_Mask_Nodes)
            target_y = y
            if target_y.dim() > 2: target_y = target_y.squeeze(-1)

            acc = (stacked_pred == target_y).sum().item() / target_y.numel()

        else:
            output = model(X, edge_index)
            loss = criterion(output[:, mask], y)
            y_pred = torch.sigmoid(output[:, mask]) > 0.5
            acc = (y_pred == y).sum().item() / (y.shape[0] * y.shape[1])

    return loss.item(), acc  # loss 可能是 tensor，转一下 item


def predict_prob(model, X, edge_index, rt_meas_dim=78, device='cuda'):
    model = model.to(device)
    model.eval()
    mask = model.action_mask

    # 【修复 4】确保输入在 device
    X = X.to(device)
    edge_index = edge_index.to(device)

    # 初始化 prob 在 device 上
    prob = torch.zeros((len(X), len(mask), 2), dtype=torch.float32, device=device)

    with torch.no_grad():
        if model.name == 'NN' or model.name == 'DCNN':
            # 切片并确保在 device
            input_x = X[:, mask, -rt_meas_dim:].to(device)
            output = model(input_x).squeeze(-1)  # 添加 .squeeze(-1)
            prob_1 = torch.sigmoid(output)
            prob = torch.stack([1 - prob_1, prob_1], dim=-1)  # dim=-1 以兼容 2D

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

                # 筛选并 sigmoid
                selected_out = out[mask]
                prob_1_list.append(torch.sigmoid(selected_out))

            prob_1 = torch.stack(prob_1_list, dim=0)  # (Batch, Num_Mask_Nodes)
            prob = torch.stack([1 - prob_1, prob_1], dim=2)  # (Batch, Num_Mask_Nodes, 2)
        elif model.name in ['mainModel', 'WithoutGCN', 'WithoutTransformer', 'WithoutCNN']:
            # print("============================================================进入此处了")
            prob_1_list = []
            for i in range(len(X)):
                out = model(X[i],edge_index)  # (1, N)
                if out.dim() == 2 and out.size(0) == 1:
                    out = out.squeeze(0)  # (N,)

                # 筛选并 sigmoid
                selected_out = out[mask]
                prob_1_list.append(torch.sigmoid(selected_out))

            prob_1 = torch.stack(prob_1_list, dim=0)  # (Batch, Num_Mask_Nodes)
            prob = torch.stack([1 - prob_1, prob_1], dim=2)  # (Batch, Num_Mask_Nodes, 2)

        else:
            output = model(X, edge_index)
            prob_1 = torch.sigmoid(output)[:, mask]
            prob = torch.stack([1 - prob_1, prob_1], dim=2)

    return prob


# evaluate_acc 和 evaluate_performance 基本不需要大改，因为它们调用 predict_prob
# 但为了安全，也可以加上 .to(device) 检查，不过只要 predict_prob 处理了，这里通常没问题
def evaluate_acc(model, X, y, edge_index, device='cuda'):
    prob = predict_prob(model, X, edge_index, device=device)
    y = y.to(device)  # 确保 y 在 device
    pred = torch.argmax(prob, dim=2)
    accuracy = (pred == y).sum().item() / (y.shape[0] * y.shape[1])
    return accuracy


def evaluate_performance(models, X, y, edge_index, device='cuda'):
    metrics = []
    # 确保全局数据在 device
    X = X.to(device)
    y = y.to(device)
    edge_index = edge_index.to(device)

    for name, model in models.items():
        model.eval()
        prob = predict_prob(model, X, edge_index, device=device)
        pred_ts = torch.argmax(prob, dim=2)

        # 确保 y 的形状和 pred_ts 一致
        # 注意：如果 y 是全节点标签，而 pred_ts 是 mask 后的，这里需要对应
        # 根据前面的逻辑，predict_prob 返回的是 mask 后的结果 (len(mask))
        # 所以 y 也应该取 mask 部分，或者在生成数据集时就已经切好了
        # 假设这里的 y 传入时已经是 mask 过的，或者形状匹配

        y_target = y  # 如果 y 是全图，这里可能需要 y[:, mask]

        accuracy = (pred_ts == y_target).sum().item() / (y_target.shape[0] * y_target.shape[1])

        # 转回 CPU 进行 sklearn 计算 (sklearn 不支持 GPU tensor)
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