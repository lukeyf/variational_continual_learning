import numpy as np
import torch
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from .processing import *

def train(model, trainloader, optimizer, epoch, device, kl_weight=1, task_id=0, binary_label=None, use_prior=False):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)  # Flatten the images if necessary
        
        optimizer.zero_grad()
        
        output = model(data, sample=True, task_id=task_id)
        if binary_label is not None:
            target = (target == binary_label[0]).long()
        
        reconstruction_loss = F.cross_entropy(output, target, reduction='mean')
        
        # Adjust loss based on whether to include KL divergence
        if task_id == 0 and not use_prior:
            loss = reconstruction_loss
        else:
            kl_divergence = model.kl_divergence()
            loss = reconstruction_loss + kl_weight * kl_divergence
            
        loss.backward()
        optimizer.step()
        
        # Optionally, uncomment the next line to log training progress
        # print(f"Epoch: {epoch}, Batch: {batch_idx+1}/{len(trainloader)}, Loss: {loss.item():.4f}")

def test(model, testloader, device, task_id=0, binary_label=None):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)  # Flatten the images if necessary
            
            output = model(data, sample=False, task_id=task_id)
            if binary_label is not None:
                target = (target == binary_label[0]).long()

            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testloader.dataset)
    
    # Optionally, uncomment the next line to log testing results
    # print(f'\nTest Set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(testloader.dataset)} ({100. * correct / len(testloader.dataset):.0f}%)\n')
    
    return test_loss, correct / len(testloader.dataset)


def run_vcl(model, train_loaders, test_loaders, optimizer, epoch_per_task, coreset_size=0, beta=1, binary_labels=None, return_individual_acc=False, use_prior=False, device='cpu'):
    ave_acc_trend_rc = []
    prev_test_loaders = []
    coresets = []
    individual_accuracies = []
    binary_labels = binary_labels if binary_labels is not None else [None] * model.num_tasks

    for task_id, (train_loader, test_loader) in enumerate(zip(train_loaders, test_loaders)):
        task_accuracies_rc = []
        if coreset_size > 0:
            for i in range(len(coresets)):
                coreset_loader = DataLoader(coresets[i], batch_size=64, shuffle=True)  # Assuming batch_size is 64
                for epoch in range(1, epoch_per_task + 1):
                    train(model, coreset_loader, optimizer, epoch, device, beta, task_id=i, binary_label=binary_labels[i], use_prior=use_prior)
                model.update_priors()

        for epoch in range(1, epoch_per_task + 1):
            train(model, train_loader, optimizer, epoch, device, beta, task_id=task_id, binary_label=binary_labels[task_id], use_prior=use_prior)
        model.update_priors()

        prediction_model = model.__class__(model.input_size, model.hidden_sizes, model.output_size, model.num_tasks, model.single_head).to(device)
        prediction_model.load_state_dict(model.state_dict())

        if coreset_size > 0:
            coresets.append(random_coreset(train_loader.dataset, coreset_size))
            for i in range(len(coresets)):
                coreset_loader = DataLoader(coresets[i], batch_size=64, shuffle=True)  # Assuming batch_size is 64
                for epoch in range(1, epoch_per_task + 1):
                    train(prediction_model, coreset_loader, optimizer, epoch, device, beta, task_id=i, binary_label=binary_labels[i], use_prior=use_prior)

        prev_test_loaders.append(test_loader)
        for task_num, ptl in enumerate(prev_test_loaders):
            test_loss, task_accuracy = test(prediction_model, ptl, device, task_id=task_num, binary_label=binary_labels[task_num])
            task_accuracies_rc.append(task_accuracy)

        average_accuracy = sum(task_accuracies_rc) / len(task_accuracies_rc)
        ave_acc_trend_rc.append(average_accuracy)
        individual_accuracies.append(task_accuracies_rc)
        print(f'Average Accuracy across {len(task_accuracies_rc)} tasks: {average_accuracy*100:.2f}%')

    if return_individual_acc:
        return ave_acc_trend_rc, individual_accuracies
    return ave_acc_trend_rc

def scale_similarity(sim, a, b):
    return 1 / (1 + np.exp(-20 * (sim - (a + b) / 2)))

def run_auto_vcl(model, train_loaders, test_loaders, optimizer, epoch_per_task, coreset_size, beta_star=1, raw_training_epoch=1, raw_train_size=1000, binary_labels=None, dor=False, return_betas=False, device='cuda'):
    task_difficulties, ave_acc_trend_rc, prev_test_loaders, coresets, betas, diff_gaps = [], [], [], [], [], []
    binary_labels = binary_labels if binary_labels is not None else [None] * model.output_size
    batch_size = 256  

    for task_id, (train_loader, test_loader) in enumerate(zip(train_loaders, test_loaders)):
        raw_acc = []
        for _ in range(10):
            raw_model = model.__class__(model.input_size, model.hidden_sizes, model.output_size, model.num_tasks, model.single_head).to(device)
            raw_trainset = random_coreset(train_loader.dataset, raw_train_size)
            raw_train_loader = DataLoader(raw_trainset, batch_size=batch_size, shuffle=True)
            raw_optimizer = Adam(raw_model.parameters(), lr=0.001)

            for epoch in range(1, raw_training_epoch + 1):
                train(raw_model, raw_train_loader, raw_optimizer, epoch, device, beta_star, task_id=0, binary_label=binary_labels[task_id])
            _, acc_simple_train = test(raw_model, test_loader, device, task_id=0, binary_label=binary_labels[task_id])
            raw_acc.append(acc_simple_train)
        
        acc_simple_train = np.mean(raw_acc)
        dummy_pred = 1 / model.output_size
        curr_difficulty = np.clip((1 - (acc_simple_train - dummy_pred) / (1 - dummy_pred)), 0, 1)

        if task_id > 0:
            _, raw_pred = test(model, test_loader, device, task_id=task_id-1, binary_label=binary_labels[task_id])
            similarity = scale_similarity(np.abs(raw_pred - dummy_pred), 0, 1 - dummy_pred)
            prev_difficulty = np.max(task_difficulties)
            diff_gaps.append(np.abs(prev_difficulty - curr_difficulty))
            avg_diff_gaps = np.mean(diff_gaps)
            beta = beta_star * np.exp((prev_difficulty - curr_difficulty / (1 + avg_diff_gaps * task_id)) * 5 + similarity * 5)
            betas.append(beta)
        else:
            beta = beta_star

        if coreset_size > 0:
            coresets.append(random_coreset(train_loader.dataset, coreset_size))
            replay_coresets = coresets
            if dor and task_id > 0:
                zipped_and_indices = sorted(enumerate(zip(task_difficulties, coresets)), key=lambda x: x[1][0], reverse=True)
                sorted_task_nums = [index for index, _ in zipped_and_indices]
                replay_coresets = [pair[1] for _, pair in zipped_and_indices]
                replay_betas = [beta_star * 10 ** (2 - d) for d, _ in sorted(zip(task_difficulties, coresets), key=lambda x: x[0], reverse=True)]

            for i, coreset in enumerate(replay_coresets):
                coreset_loader = DataLoader(coreset, batch_size=batch_size, shuffle=True)
                replay_beta = replay_betas[i] if dor else beta
                replay_task_id = sorted_task_nums[i] if dor else i
                for epoch in range(1, epoch_per_task + 1):
                    train(model, coreset_loader, optimizer, epoch, device, replay_beta, task_id=replay_task_id, binary_label=binary_labels[replay_task_id])
            model.update_priors()

        for epoch in range(1, epoch_per_task + 1):
            train(model, train_loader, optimizer, epoch, device, beta, task_id=task_id, binary_label=binary_labels[task_id])
        model.update_priors()

        prediction_model = model.__class__(model.input_size, model.hidden_sizes, model.output_size, model.num_tasks, model.single_head).to(device)
        prediction_model.load_state_dict(model.state_dict())

        task_difficulties.append(curr_difficulty)
        prev_test_loaders.append(test_loader)
        task_accuracies_rc = [test(prediction_model, ptl, device, task_id=i, binary_label=binary_labels[i])[1] for i, ptl in enumerate(prev_test_loaders)]
        ave_acc_trend_rc.append(np.mean(task_accuracies_rc))
        print(f'Average Accuracy across {len(task_accuracies_rc)} tasks: {np.mean(task_accuracies_rc) * 100:.2f}%')

    return (ave_acc_trend_rc, betas) if return_betas else ave_acc_trend_rc
