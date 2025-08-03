for batch in data_loader:

    optimizer.zero_grad()

    total_loss = 0.0
    for microbatch in batch:
        loss = model.forward(microbatch)
        loss.backward()
        total_loss += loss.detach().item() 

    optimizer.step()
    scheduler.step()