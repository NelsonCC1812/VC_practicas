class EarlyStopper:
    
    def __init__(self, patience=5, min_delta=0, callback=lambda **_: None):
        self.patience = patience
        self.min_delta = min_delta
        self.callback = callback

        self.counter = 1

        self.best_model = None
        self.best_loss = float("inf")

    def __call__(self, val_loss, model):

        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0

            self.best_model = model.state_dict()
            return False

        self.counter += 1
        if self.counter >= self.patience:
            self.callback(counter=self.counter, best_loss=self.best_loss, best_model=self.best_model)
            return True

    def reset(self):
        self.early_stop = False
        self.counter = 1