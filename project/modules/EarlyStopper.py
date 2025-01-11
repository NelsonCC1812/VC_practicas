class EarlyStopper:
    
    def __init__(self, patience=5, min_delta=0, verbose=False, callback=lambda **_: None):
        self.patience = patience
        self.min_delta = min_delta
        self.callback = callback
        self.verbose = verbose
        
        self.counter = 1

        self.best_model = None
        self.best_loss = float("inf")
        self.best_epoch = None

    def __call__(self, val_loss, model,*, epoch=None):

        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 1

            self.best_model = model.state_dict()
            
            if self.verbose: print('Best model currently!!')
            return False
        if self.verbose: print(f'BEST:{f" epoch: {self.best_epoch} " if epoch else ''} val_loss: {self.best_loss}') 
        self.counter += 1
        if self.counter >= self.patience:
            self.callback(counter=self.counter, best_loss=self.best_loss, best_model=self.best_model)
            return True

    def reset(self):
        self.counter = 1