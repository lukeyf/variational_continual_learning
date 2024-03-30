

def run_vcl(model, train_loaders, test_loaders, optimizer, epoch_per_task, coreset_size=0, beta=1, binary_labels = None):
    ave_acc_trend_rc = []
    prev_test_loaders= []
    coresets = []
    if binary_labels is None:
        binary_labels = [None] * model.output_size
    for task_id, (train_loader, test_loader) in enumerate(zip(train_loaders, test_loaders), start=0):
        task_accuracies_rc = []
        if coreset_size > 0:
            for i in (range(len(coresets))):
                for epoch in (range(1, epoch_per_task + 1)):
                    coreset_loader = DataLoader(coresets[i], batch_size=batch_size, shuffle=True)
                    train(model, coreset_loader, optimizer, epoch, device, beta, task_id=i, binary_label=binary_labels[i])
                model.update_priors()
        for epoch in (range(1, epoch_per_task + 1)):
            train(model, train_loader, optimizer, epoch, device, beta, task_id=task_id, binary_label=binary_labels[task_id])
        model.update_priors()


        # for prediction
        prediction_model = type(model)(model.input_size, model.hidden_sizes, model.output_size, model.num_tasks).to(device)
        prediction_model.load_state_dict(model.state_dict())
        # replay
        if coreset_size > 0:
            coresets.append(random_coreset(train_loader.dataset, coreset_size))
            for i in (range(len(coresets))):
                for epoch in (range(1, epoch_per_task + 1)):
                    coreset_loader = DataLoader(coresets[i], batch_size=batch_size, shuffle=True)
                    train(prediction_model, coreset_loader, optimizer, epoch, device, beta, task_id=i, binary_label=binary_labels[i])
        task_num = 0  
        prev_test_loaders.append(test_loader)
        for ptl in prev_test_loaders: 
            test_loss, task_accuracy = test(prediction_model, ptl, device,task_id=task_num, binary_label=binary_labels[task_num])
            task_accuracies_rc.append(task_accuracy)
            task_num += 1
        average_accuracy = sum(task_accuracies_rc) / len(task_accuracies_rc)
        ave_acc_trend_rc.append(average_accuracy)
        print(f'Average Accuracy across {len(task_accuracies_rc)} tasks: {average_accuracy*100:.2f}%')
    return ave_acc_trend_rc