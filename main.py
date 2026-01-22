# 标准库
import argparse
import os
import sys
import time
from distutils.util import strtobool # type: ignore

# 第三方库
import requests
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from medmnist import INFO, Evaluator
from sklearn.metrics import (confusion_matrix, f1_score,
                                 precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import label_binarize

# 本地模块
from MedViT import MedViT_base, MedViT_large, MedViT_small, MedViT_tiny
from cedar.utils import print,create_name
from datasets import build_dataset
#from MedViTV1 import MedViT_small, MedViT_base, MedViT_large
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(ROOT_DIR, 'logs', create_name()+'.log')
os.environ['ROOT_DIR'] = ROOT_DIR
os.environ['LOG_PATH'] = LOG_PATH
print(f"ROOT_DIR: {ROOT_DIR}")
print(f"LOG_PATH: {LOG_PATH}")

model_classes = {
    'MedViT_tiny': MedViT_tiny,
    'MedViT_small': MedViT_small,
    'MedViT_base': MedViT_base,
    'MedViT_large': MedViT_large
}

model_urls = {
    "MedViT_tiny": "https://dl.dropbox.com/scl/fi/496jbihqp360jacpji554/MedViT_tiny.pth?rlkey=6hb9froxugvtg8l639jmspxfv&st=p9ef06j8&dl=0",
    "MedViT_small": "https://dl.dropbox.com/scl/fi/6nnec8hxcn5da6vov7h2a/MedViT_small.pth?rlkey=yf5twra1cv6ep2oqr79tbzyg5&st=rwx5hy8z&dl=0",
    "MedViT_base": "https://dl.dropbox.com/scl/fi/q5c0u515dd4oc8j55bhi9/MedViT_base.pth?rlkey=5duw3uomnsyjr80wykvedjhas&st=incconx4&dl=0",
    "MedViT_large": "https://dl.dropbox.com/scl/fi/owujijpsl6vwd481hiydd/MedViT_large.pth?rlkey=cx9lqb4a1288nv4xlmux13zoe&st=kcehwbrb&dl=0"
}

def download_checkpoint(url, path):
    print(f"Downloading checkpoint from {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
    print(f"Checkpoint downloaded and saved to {path}")

def topk(output, target, k=(1,)):
    """Calculate top-k accuracy.
    
    Args:
        output: Model output tensor of shape (batch_size, num_classes)
        target: Ground truth labels tensor of shape (batch_size,)
        k: Tuple of k values to calculate (e.g., (1, 5) for top-1 and top-5)
    
    Returns:
        List of top-k accuracies for each k value
    """
    with torch.no_grad():
        maxk = max(k)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k_val in k:
            correct_k = correct[:k_val].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res

# Define the MNIST training routine
def train_mnist(epochs, net, train_loader, test_loader, optimizer, scheduler, loss_function, device, save_path, data_flag, task):
    best_acc = 0.0
    total_steps = epochs * len(train_loader)
    start_time = time.time()
    step_times = []
    
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        epoch_start_time = time.time()
        epoch_data_times = []
        epoch_forward_times = []
        epoch_backward_times = []
        
        for step, datax in enumerate(train_loader):
            step_start_time = time.time()
            
            # Data loading time (already loaded by DataLoader, measure transfer time)
            data_start = time.time()
            images, labels = datax
            images, labels = images.to(device), labels.to(device)
            data_time = time.time() - data_start
            epoch_data_times.append(data_time)
            
            # Forward pass
            forward_start = time.time()
            optimizer.zero_grad()
            outputs = net(images)
            
            if task == 'multi-label, binary-class':
                labels = labels.to(torch.float32)
                loss = loss_function(outputs, labels)
            else:
                labels = labels.squeeze().long()
                loss = loss_function(outputs.squeeze(0), labels)
            forward_time = time.time() - forward_start
            epoch_forward_times.append(forward_time)
            
            # Backward pass
            backward_start = time.time()
            loss.backward()
            optimizer.step()
            scheduler.step()
            backward_time = time.time() - backward_start
            epoch_backward_times.append(backward_time)
            
            running_loss += loss.item()
            step_time = time.time() - step_start_time
            step_times.append(step_time)
            
            # Keep only last 50 step times for ETA calculation
            if len(step_times) > 50:
                step_times.pop(0)
            
            # Calculate progress and ETA using actual elapsed time for better accuracy
            current_step = epoch * len(train_loader) + step + 1
            progress = (current_step / total_steps) * 100
            
            # Use actual elapsed time for more accurate ETA
            elapsed_time = time.time() - start_time
            if current_step > 0:
                avg_time_per_step = elapsed_time / current_step
            else:
                avg_time_per_step = sum(step_times) / len(step_times) if step_times else 0.0
            
            # Also keep recent step average as fallback
            recent_avg_step_time = sum(step_times) / len(step_times) if step_times else 0.0
            # Use weighted average: 70% actual elapsed time, 30% recent average (adapts to current speed)
            avg_step_time = 0.7 * avg_time_per_step + 0.3 * recent_avg_step_time
            
            remaining_steps = total_steps - current_step
            eta_seconds = remaining_steps * avg_step_time
            eta_hours = int(eta_seconds // 3600)
            eta_minutes = int((eta_seconds % 3600) // 60)
            eta_secs = int(eta_seconds % 60)
            
            # Calculate epoch-level progress and ETA
            current_epoch = epoch + 1
            epoch_progress = (step + 1) / len(train_loader) * 100
            epoch_elapsed_time = time.time() - epoch_start_time
            if step > 0:
                epoch_avg_time_per_step = epoch_elapsed_time / (step + 1)
            else:
                epoch_avg_time_per_step = recent_avg_step_time
            # Use weighted average for epoch ETA too
            epoch_avg_step_time = 0.7 * epoch_avg_time_per_step + 0.3 * recent_avg_step_time
            epoch_remaining_steps = len(train_loader) - step - 1
            epoch_eta_seconds = epoch_remaining_steps * epoch_avg_step_time
            epoch_eta_minutes = int(epoch_eta_seconds // 60)
            epoch_eta_secs = int(epoch_eta_seconds % 60)
            
            # Get learning rate
            current_lr = scheduler.get_last_lr()[0]
            
            # Get GPU memory usage (reset at start of next step)
            if torch.cuda.is_available():
                max_mem = torch.cuda.max_memory_allocated(device) / 1024 / 1024  # MB
                if (step + 1) % 10 == 0 or (step + 1) == len(train_loader):
                    torch.cuda.reset_peak_memory_stats(device)
            else:
                max_mem = 0
            
            if (step + 1) % 10 == 0 or (step + 1) == len(train_loader):
                print(f"epoch: [{current_epoch}/{epochs}]  "
                      f"step: [{step+1}/{len(train_loader)}] ({epoch_progress:.1f}%)  "
                      f"eta_epoch: {epoch_eta_minutes}:{epoch_eta_secs:02d}  "
                      f"eta_total: {eta_hours}:{eta_minutes:02d}:{eta_secs:02d}  "
                      f"loss: {loss.item():.4f}  "
                      f"time: {step_time:.3f}  data_time: {data_time:.3f}  " # 增加iter: {current_step}
                      f"forward_time: {forward_time:.3f}  backward_time: {backward_time:.3f}  " 
                      f"lr: {current_lr:.6f}  max_mem: {max_mem:.0f}M  "
                      )
        
        # Epoch performance statistics
        epoch_time = time.time() - epoch_start_time
        avg_data_time = sum(epoch_data_times) / len(epoch_data_times)
        avg_forward_time = sum(epoch_forward_times) / len(epoch_forward_times)
        avg_backward_time = sum(epoch_backward_times) / len(epoch_backward_times)
        avg_step_time_epoch = sum(step_times[-len(train_loader):]) / min(len(train_loader), len(step_times))
        samples_per_sec = len(train_loader.dataset) / epoch_time
        
        print(f"[epoch {epoch + 1} training completed] "
              f"avg_loss: {running_loss / len(train_loader):.4f}  "
              f"epoch_time: {epoch_time:.2f}s  "
              f"avg_data_time: {avg_data_time:.3f}s  "
              f"avg_forward_time: {avg_forward_time:.3f}s  "
              f"avg_backward_time: {avg_backward_time:.3f}s  "
              f"avg_step_time: {avg_step_time_epoch:.3f}s  "
              f"throughput: {samples_per_sec:.1f} samples/s")
        
        net.eval()
        y_score = torch.tensor([])
        top1_list = []
        top5_list = []
        with torch.no_grad():
            print(f"Validating epoch[{epoch + 1}/{epochs}]...")
            for val_data in test_loader:
                inputs, targets = val_data
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                
                if task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32)
                    outputs_softmax = outputs.softmax(dim=-1)
                    # For multi-label, use argmax of targets as ground truth
                    targets_for_acc = targets.argmax(dim=1) if len(targets.shape) > 1 else targets.long()
                else:
                    targets = targets.squeeze().long()
                    outputs_softmax = outputs.softmax(dim=-1)
                    targets_for_acc = targets
                    targets_resized = targets.float().resize_(len(targets), 1)
                
                # Calculate top-k accuracy for this batch
                num_classes = outputs.size(1)
                if num_classes >= 5:
                    tops = topk(outputs, targets_for_acc, k=(1, 5))
                    top1_list.append(tops[0])
                    top5_list.append(tops[1])
                else:
                    tops = topk(outputs, targets_for_acc, k=(1,))
                    top1_list.append(tops[0])
                    top5_list.append(tops[0])  # If fewer than 5 classes, top5 = top1
                
                y_score = torch.cat((y_score, outputs_softmax.cpu()), 0)
                
        y_score = y_score.detach().numpy()
        evaluator = Evaluator(data_flag, 'test', size=224, root='./data')
        metrics = evaluator.evaluate(y_score)
        
        # Calculate average top1 and top5 accuracy
        # Note: Top-1 Acc uses batch-wise average, while MedMNIST Acc uses overall calculation
        # This can cause slight differences, especially when batch sizes vary
        top1_acc = sum(top1_list) / len(top1_list) / 100.0  # Convert from percentage to fraction
        top5_acc = sum(top5_list) / len(top5_list) / 100.0  # Convert from percentage to fraction
        
        val_accurate, _ = metrics
        num_classes = outputs.size(1)
        print(f'\n{"="*60}')
        print(f'[Epoch {epoch + 1}/{epochs}] Validation Results:')
        print(f'  Train Loss:    {running_loss / len(train_loader):.4f}')
        print(f'  AUC:           {metrics[0]:.4f}')
        print(f'  Top-1 Acc:      {top1_acc:.4f} ({top1_acc*100:.2f}%) [batch-wise avg, using logits]')
        if num_classes >= 5:
            print(f'  Top-5 Acc:      {top5_acc:.4f} ({top5_acc*100:.2f}%)')
        print(f'  MedMNIST Acc:   {metrics[1]:.4f} ({metrics[1]*100:.2f}%) [overall calc, using softmax probs]')
        print(f'{"="*60}\n')
        #print(f'lr: {scheduler.get_last_lr()[-1]:.8f}')
        if val_accurate > best_acc:
            print('\nSaving checkpoint...')
            best_acc = val_accurate
            state = {
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'acc': best_acc,
                'epoch': epoch,
            }
            torch.save(state, save_path)

    print('Finished Training')

# Define the non-MNIST training routine
def specificity_per_class(conf_matrix):
    """Calculates specificity for each class."""
    specificity = []
    for i in range(len(conf_matrix)):
        tn = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
        fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
        specificity.append(tn / (tn + fp))
    return specificity

def overall_accuracy(conf_matrix):
    """Calculates overall accuracy for multi-class."""
    tp_tn_sum = conf_matrix.trace()  # Sum of all diagonal elements (TP for all classes)
    total_sum = conf_matrix.sum()  # Sum of all elements in the matrix
    return tp_tn_sum / total_sum

def train_other(epochs, net, train_loader, test_loader, optimizer, scheduler, loss_function, device, save_path):
    best_acc = 0.0
    total_steps = epochs * len(train_loader)
    start_time = time.time()
    step_times = []
    
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        epoch_start_time = time.time()
        epoch_data_times = []
        epoch_forward_times = []
        epoch_backward_times = []

        # Training Loop
        for step, datax in enumerate(train_loader):
            step_start_time = time.time()
            
            # Data loading time (already loaded by DataLoader, measure transfer time)
            data_start = time.time()
            images, labels = datax
            images, labels = images.to(device), labels.to(device)
            data_time = time.time() - data_start
            epoch_data_times.append(data_time)
            
            # Forward pass
            forward_start = time.time()
            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            forward_time = time.time() - forward_start
            epoch_forward_times.append(forward_time)
            
            # Backward pass
            backward_start = time.time()
            loss.backward()
            optimizer.step()
            scheduler.step()
            backward_time = time.time() - backward_start
            epoch_backward_times.append(backward_time)
            
            running_loss += loss.item()
            step_time = time.time() - step_start_time
            step_times.append(step_time)
            
            # Keep only last 50 step times for ETA calculation
            if len(step_times) > 50:
                step_times.pop(0)
            
            # Calculate progress and ETA using actual elapsed time for better accuracy
            current_step = epoch * len(train_loader) + step + 1
            progress = (current_step / total_steps) * 100
            
            # Use actual elapsed time for more accurate ETA
            elapsed_time = time.time() - start_time
            if current_step > 0:
                avg_time_per_step = elapsed_time / current_step
            else:
                avg_time_per_step = sum(step_times) / len(step_times) if step_times else 0.0
            
            # Also keep recent step average as fallback
            recent_avg_step_time = sum(step_times) / len(step_times) if step_times else 0.0
            # Use weighted average: 70% actual elapsed time, 30% recent average (adapts to current speed)
            avg_step_time = 0.7 * avg_time_per_step + 0.3 * recent_avg_step_time
            
            remaining_steps = total_steps - current_step
            eta_seconds = remaining_steps * avg_step_time
            eta_hours = int(eta_seconds // 3600)
            eta_minutes = int((eta_seconds % 3600) // 60)
            eta_secs = int(eta_seconds % 60)
            
            # Calculate epoch-level progress and ETA
            current_epoch = epoch + 1
            epoch_progress = (step + 1) / len(train_loader) * 100
            epoch_elapsed_time = time.time() - epoch_start_time
            if step > 0:
                epoch_avg_time_per_step = epoch_elapsed_time / (step + 1)
            else:
                epoch_avg_time_per_step = recent_avg_step_time
            # Use weighted average for epoch ETA too
            epoch_avg_step_time = 0.7 * epoch_avg_time_per_step + 0.3 * recent_avg_step_time
            epoch_remaining_steps = len(train_loader) - step - 1
            epoch_eta_seconds = epoch_remaining_steps * epoch_avg_step_time
            epoch_eta_minutes = int(epoch_eta_seconds // 60)
            epoch_eta_secs = int(epoch_eta_seconds % 60)
            
            # Get learning rate
            current_lr = scheduler.get_last_lr()[0]
            
            # Get GPU memory usage (reset at start of next step)
            if torch.cuda.is_available():
                max_mem = torch.cuda.max_memory_allocated(device) / 1024 / 1024  # MB
                if (step + 1) % 10 == 0 or (step + 1) == len(train_loader):
                    torch.cuda.reset_peak_memory_stats(device)
            else:
                max_mem = 0
            
            if (step + 1) % 10 == 0 or (step + 1) == len(train_loader):
                print(f"epoch: [{current_epoch}/{epochs}]  "
                      f"step: [{step+1}/{len(train_loader)}] ({epoch_progress:.1f}%)  "
                      f"eta_epoch: {epoch_eta_minutes}:{epoch_eta_secs:02d}  "
                      f"eta_total: {eta_hours}:{eta_minutes:02d}:{eta_secs:02d}  "
                      f"loss: {loss.item():.4f}  "
                      f"time: {step_time:.3f}  data_time: {data_time:.3f}  "
                      f"forward_time: {forward_time:.3f}  backward_time: {backward_time:.3f}  "
                      f"lr: {current_lr:.6f}  max_mem: {max_mem:.0f}M  "
                      )
        
        # Epoch performance statistics
        epoch_time = time.time() - epoch_start_time
        avg_data_time = sum(epoch_data_times) / len(epoch_data_times)
        avg_forward_time = sum(epoch_forward_times) / len(epoch_forward_times)
        avg_backward_time = sum(epoch_backward_times) / len(epoch_backward_times)
        avg_step_time_epoch = sum(step_times[-len(train_loader):]) / min(len(train_loader), len(step_times))
        samples_per_sec = len(train_loader.dataset) / epoch_time
        
        print(f"[epoch {epoch + 1} training completed] "
              f"avg_loss: {running_loss / len(train_loader):.4f}  "
              f"epoch_time: {epoch_time:.2f}s  "
              f"avg_data_time: {avg_data_time:.3f}s  "
              f"avg_forward_time: {avg_forward_time:.3f}s  "
              f"avg_backward_time: {avg_backward_time:.3f}s  "
              f"avg_step_time: {avg_step_time_epoch:.3f}s  "
              f"throughput: {samples_per_sec:.1f} samples/s")
        
        # Validation Loop
        net.eval()
        all_preds = []
        all_labels = []
        all_probs = []  # Store raw probabilities/logits for AUC
        top1_list = []
        top5_list = []
        
        with torch.no_grad():
            print(f"Validating epoch[{epoch + 1}/{epochs}]...")
            for val_data in test_loader:
                val_images, val_labels = val_data
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                outputs = net(val_images)  # Raw outputs (logits)
                probs = torch.softmax(outputs, dim=1)  # Convert to probabilities
                
                predict_y = torch.max(probs, dim=1)[1]  # Predicted class

                # Collect predictions, labels, and probabilities
                all_preds.extend(predict_y.cpu().numpy())
                all_labels.extend(val_labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                # Calculate top-k accuracy for this batch
                num_classes = outputs.size(1)
                if num_classes >= 5:
                    tops = topk(outputs, val_labels, k=(1, 5))
                    top1_list.append(tops[0])
                    top5_list.append(tops[1])
                else:
                    tops = topk(outputs, val_labels, k=(1,))
                    top1_list.append(tops[0])
                    top5_list.append(tops[0])  # If fewer than 5 classes, top5 = top1
        
        # Calculate average top1 and top5 accuracy
        top1_acc = sum(top1_list) / len(top1_list) / 100.0  # Convert from percentage to fraction
        top5_acc = sum(top5_list) / len(top5_list) / 100.0  # Convert from percentage to fraction
        
        # Calculate metrics
        val_accurate = top1_acc
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')  # Sensitivity
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # Confusion Matrix for multi-class
        conf_matrix = confusion_matrix(all_labels, all_preds)
        specificity = specificity_per_class(conf_matrix)  # List of specificities per class
        avg_specificity = sum(specificity) / len(specificity)  # Average specificity

        # Overall Accuracy calculation
        overall_acc = overall_accuracy(conf_matrix)

        # One-hot encode the labels for AUC calculation
        n_classes = len(conf_matrix)
        all_labels_one_hot = label_binarize(all_labels, classes=list(range(n_classes)))

        try:
            # Compute AUC for multi-class
            auc = roc_auc_score(all_labels_one_hot, all_probs, multi_class='ovr')
        except ValueError:
            auc = float('nan')  # Handle edge case where AUC can't be computed

        # Get num_classes from confusion matrix
        num_classes = len(conf_matrix)
        
        # Print metrics with improved format
        print(f'\n{"="*60}')
        print(f'[Epoch {epoch + 1}/{epochs}] Validation Results:')
        print(f'  Train Loss:     {running_loss / len(train_loader):.4f}')
        print(f'  Top-1 Acc:       {top1_acc:.4f} ({top1_acc*100:.2f}%)')
        if num_classes >= 5:
            print(f'  Top-5 Acc:       {top5_acc:.4f} ({top5_acc*100:.2f}%)')
        print(f'  Precision:      {precision:.4f}')
        print(f'  Recall:          {recall:.4f}')
        print(f'  Specificity:     {avg_specificity:.4f}')
        print(f'  F1-Score:        {f1:.4f}')
        print(f'  AUC:             {auc:.4f}')
        print(f'  Overall Acc:     {overall_acc:.4f}')
        print(f'{"="*60}\n')
        
        #print(f'lr: {scheduler.get_last_lr()[-1]:.8f}')
        
        # Save best model
        if val_accurate > best_acc:
            print('\nSaving checkpoint...')
            best_acc = val_accurate
            state = {
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'acc': best_acc,
                'epoch': epoch,
            }
            torch.save(state, save_path)

    print('Finished Training')

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))
    
    # 打印配置参数
    print("\n========== Configuration Parameters ==========")
    print(f"model_name: {args.model_name}")
    print(f"dataset: {args.dataset}")
    print(f"pretrained: {args.pretrained}")
    print(f"batch_size: {args.batch_size}")
    print(f"lr: {args.lr}")
    print(f"epochs: {args.epochs}")
    print("============================================\n")
    
    model_name = args.model_name
    dataset_name = args.dataset
    pretrained = args.pretrained
    if args.dataset.endswith('mnist'):
        info = INFO[args.dataset]
        task = info['task']
        if task == "multi-label, binary-class":
            loss_function = nn.BCEWithLogitsLoss()
        else:
            loss_function = nn.CrossEntropyLoss()
    else:
        loss_function = nn.CrossEntropyLoss()
    model_class = model_classes.get(model_name)

    # if not model_class:
    #     raise ValueError(f"Model {model_name} is not recognized. Available models: {list(model_classes.keys())}")

    batch_size = args.batch_size
    lr = args.lr
    
    train_dataset, test_dataset, nb_classes = build_dataset(args=args)
    val_num = len(test_dataset)
    train_num = len(train_dataset)
    
    # scheduler max iteration
    eta = args.epochs * train_num // args.batch_size

    # Select model
    if model_name in model_classes:
        model_class = model_classes[model_name]
        net = model_class(num_classes=nb_classes).cuda()
        if pretrained:
            checkpoint_path = args.checkpoint_path
            if not os.path.exists(checkpoint_path):
                checkpoint_url = model_urls.get(model_name)
                if not checkpoint_url:
                    raise ValueError(f"Checkpoint URL for model {model_name} not found.")
                download_checkpoint(checkpoint_url, f'./{model_name}.pth')
                checkpoint_path = f'./{model_name}.pth'

            checkpoint = torch.load(checkpoint_path)
            state_dict = net.state_dict()
            for k in ['proj_head.0.weight', 'proj_head.0.bias']:
                if k in checkpoint and checkpoint[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint[k]
            net.load_state_dict(checkpoint, strict=False)
    else:
        net = timm.create_model(model_name, pretrained=pretrained, num_classes=nb_classes).cuda()

    
    optimizer = optim.AdamW(net.parameters(), lr=lr, betas=[0.9, 0.999], weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=eta, eta_min=5e-6)
    
    # Optimize DataLoader with multi-process loading and pinned memory
    num_workers = min(4, os.cpu_count() or 1)  # Use 4 workers or available CPU cores, whichever is smaller
    train_loader = data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    test_loader = data.DataLoader(
        dataset=test_dataset, 
        batch_size=2*batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    print(train_dataset)
    print("===================")
    print(test_dataset)

    epochs = args.epochs
    best_acc = 0.0
    save_path = f'./{model_name}_{dataset_name}.pth'
    train_steps = len(train_loader)

    if dataset_name.endswith('mnist'):
        
        train_mnist(epochs, net, train_loader, test_loader,
        optimizer, scheduler, loss_function, device, save_path, dataset_name, task)
    else:
        train_other(epochs, net, train_loader, test_loader,
        optimizer, scheduler, loss_function, device, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script for MedViT models.')
    parser.add_argument('--model_name', type=str, default='MedViT_tiny', help='Model name to use.')
    #tissuemnist, pathmnist, chestmnist, dermamnist, octmnist, pneumoniamnist, retinamnist, breastmnist, bloodmnist,
    #organamnist, organcmnist, organsmnist'
    parser.add_argument('--dataset', type=str, default='PAD', help='Dataset to use.')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--pretrained', type=lambda x: bool(strtobool(x)), default=False, help="Whether to use pretrained weights (True/False).")
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/MedViT_tiny.pth', help='Path to the checkpoint file.')

    args = parser.parse_args()
    main(args)

# python main.py --model_name 'convnext_tiny' --dataset 'PAD'
