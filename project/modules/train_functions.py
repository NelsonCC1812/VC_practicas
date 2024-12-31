import torch
from tqdm import tqdm

# ---
import sys; sys.path.append('../')
from commons.HistCollection import *



def eval(model, loader, *, device, criterion, dst):
    """returns loss, posit_dst_h, negat_dst_h, asetid_h, nsetid_h
    could extract posit&negats mean, std, median, diferences...
    """
    if not (device and criterion and dst): raise Exception("Params needed")

    eval_h = HistCollection()

    loss = 0.0
    total = 0

    model.eval()

    with torch.no_grad():
        for anchs, posits, negats, asetids, nsetids  in loader:
            
            anchs, posits, negats = anchs.to(device), posits.to(device), negats.to(device)
            anchs, posits, negats = model(anchs), model(posits), model(negats)

            loss += criterion(anchs, posits, negats).item()
            total += len(asetids)

            eval_h.posit_dst.extend([dst(anchs[i], posits[i]) for i in range(len(anchs))])
            eval_h.negat_dst.extend([dst(anchs[i], negats[i]) for i in range(len(anchs))])

            eval_h.asetid.extend(asetids)
            eval_h.nsetid.extend(nsetids)


    eval_h.loss = [loss/len(loader)]
    
    return  eval_h



def train_loop(model, loader, *, optimizer, criterion, dst, num_epoch=100, 
               device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
               early_stopper, scheduler, callback=lambda **_: None,
               train_h = HistCollection(), val_h = HistCollection(), val_loader): 
    
    if not (optimizer and criterion and dst and early_stopper and scheduler): raise Exception("Params needed")


    for epoch in range(1,num_epoch+1):

        running_loss = 0.0
        total = 0

        running_posit_dst_h = []
        running_negat_dst_h = []

        running_asetid_h = []
        running_nsetid_h = []

        model.train()

        for anchs, posits, negats, asetids, nsetids in tqdm(loader):

            anchs, posits, negats = anchs.to(device), posits.to(device), negats.to(device)
            optimizer.zero_grad()

            anchs, posits, negats = model(anchs), model(posits), model(negats)
            loss = criterion(anchs, posits, negats)
            
            loss.backward()
            optimizer.step()

            # analysis ---

            running_loss += loss.item()

            running_posit_dst_h.extend([dst(anchs[i], posits[i]) for i in range(len(anchs))])
            running_negat_dst_h.extend([dst(anchs[i], negats[i]) for i in range(len(anchs))])

            running_asetid_h.extend(asetids)
            running_nsetid_h.extend(nsetids)

            total += asetids.size(0)


        # eval
        val_loss, val_posit_dst_h, val_negat_dst_h, val_asetid_h, val_nsetid_h  = eval(model, val_loader, device=device, criterion=criterion)
        scheduler.step(val_loss)

        # analysis ---

            # train
        train_loss = running_loss/len(loader)
        train_h.train.append(train_loss)

        train_h.posit_dst.append(running_posit_dst_h)
        train_h.negat_dst.append(running_negat_dst_h)

        train_h.asetid.append(running_asetid_h)
        train_h.nsetid.append(running_nsetid_h)

            # val
        val_h.loss.append(val_loss)

        val_h.posit_dst.append(val_posit_dst_h)
        val_h.negat_dst.append(val_negat_dst_h)

        val_h.asetid.append(val_asetid_h)
        val_h.nsetid.append(val_nsetid_h)

        # ---
        callback(epoch=epoch, train_loss=train_loss, val_loss=val_loss, train_h=train_h, val_h=val_h)

        if(early_stopper(val_loss, model)): break

    return train_h, val_h



def save_model(model, path, complete=False):
    try: torch.save(model if complete else model.state_dict(), path)
    except Exception as e: print("Error al guardar el model:", e)