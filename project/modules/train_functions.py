import torch
from tqdm import tqdm

def eval(model, loader, *, device, criterion, dst):
    """returns loss, posit_dst_h, negat_dst_h, asetid_h, nsetid_h
    could extract posit&negats mean, std, median, diferences
    """
    if not (device and criterion and dst): raise Exception("Params needed")

    model.eval()

    posit_dst_h = [] # distance of the anchor relative to positive
    negat_dst_h = [] # distance of the anchor relative to negative

    asetid_h = [] # setsids of the anchors
    nsetid_h = [] # setsids of the negatives

    loss = 0.0
    total = 0

    with torch.no_grad():
        for anchs, posits, negats, asetid, nsetid  in loader:
            
            anchs, posits, negats = anchs.to(device), posits.to(device), negats.to(device)
            anchs, posits, negats = model(anchs), model(posits), model(negats)

            loss += criterion(anchs, posits, negats).item()
            total += len(asetid)

            posit_dst_h.extend([dst(anchs[i], posits[i]) for i in range(len(anchs))])
            negat_dst_h.extend([dst(anchs[i], negats[i]) for i in range(len(anchs))])

            asetid_h.extend(asetid)
            nsetid_h.extend(nsetid)

    
    return  loss/len(loader), posit_dst_h, negat_dst_h, asetid_h, nsetid_h



def train_loop(model, loader, *, optimizer, criterion, dst, num_epoch=100, 
               device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
               train_loss_h=[], val_loss_h=[],
               callback=lambda **_: None, early_stopper, scheduler, val_loader):
    
    if not (optimizer and criterion and dst and early_stopper and scheduler): raise Exception("Params needed")
    
    val_preds, val_labels, val_loss, val_acc  = eval(model, val_loader, device=device, criterion=criterion)
    callback(epoch=0, val_acc=val_acc, val_loss=val_loss, train_loss=float("inf"), train_acc=0)

    for epoch in range(num_epoch):

        running_loss = 0.0
        total = 0

        model.train()

        for imgs, labels in tqdm(loader):

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
    try: torch.save(model if complete else model.state_dict(), path)
    except Exception as e: print("Error al guardar el model:", e)