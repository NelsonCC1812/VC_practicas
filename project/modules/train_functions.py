import torch

def eval(model, loader, *, device = None, criterion = None):
    """returns pred, real, accuracy, val_loss"""
    if not (device and criterion): raise Exception("Params needed")

    model.eval()

    corrects = 0

    pred = []
    real = []
    loss = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            
            imgs, labels = imgs.to(device), labels.to(device)

            outs = model(imgs)
            loss += criterion(outs.data, labels).item()

            _, predicted = torch.max(outs.data, 1)
            
            total += labels.size(0)
            pred.extend(predicted.tolist())
            real.extend(labels.tolist())

            corrects += (predicted == labels).sum().item()
    
    return  pred, real, loss/len(loader), corrects/total



def train_loop(model, loader, *, optimizer, criterion, num_epoch=100, 
               device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
               train_loss_h=[], train_acc_h=[], val_loss_h=[] ,val_acc_h=[], 
               callback=lambda **_: None, early_stopper, scheduler, val_loader):
    
    if not (optimizer and criterion and early_stopper and scheduler): raise Exception("Params needed")
    
    val_preds, val_labels, val_loss, val_acc  = eval(model, val_loader, device=device, criterion=criterion)
    callback(epoch=0, val_acc=val_acc, val_loss=val_loss, train_loss=float("inf"), train_acc=0)

    for epoch in range(num_epoch):

        running_loss = 0.0
        corrects = 0
        total = 0

        model.train()

        for i, (imgs, labels) in enumerate(loader, 1):

            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            outs = model(imgs)
            loss = criterion(outs, labels)
            
            running_loss += loss.item()
            
            loss.backward()
            optimizer.step()

            # extra for analytics
            _, predicted = torch.max(outs, 1)
            total += labels.size(0)
            corrects += (predicted == labels).sum().item()
            # ==================

            if not i % 10:
                print(f"\tEpoch {epoch+1}, Batch {i}: Loss: {running_loss*(len(loader)-i+1)/(10*len(loader)):.4f}")
            
        val_preds, val_labels, val_loss, val_acc  = eval(model, val_loader, device=device, criterion=criterion)
        scheduler.step(val_loss)
        
        # extra for analytics
        train_loss = running_loss/len(loader)
        train_loss_h.append(train_loss)
        train_acc = 100*corrects/total
        train_acc_h.append(train_acc)

        val_loss_h.append(val_loss)
        val_acc_h.append(val_acc*100)
        # ===================

        callback(epoch=epoch+1, val_acc=val_acc, val_loss=val_loss, train_loss=train_loss, train_acc=train_acc)

        if(early_stopper(val_loss, model)): break

    return train_acc_h, val_acc_h, train_loss_h, val_loss_h, val_preds, val_labels


def save_model(model, path, complete=False):
    try:
        torch.save(model if complete else model.state_dict(), path)
    except Exception as e:
        print("Error al guardar el model:", e)