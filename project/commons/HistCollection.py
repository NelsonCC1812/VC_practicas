class HistCollection():
    def __init__(self, loss=None, posit_dst=None, negat_dst=None, asetid=None, nsetid=None):
        self.loss = loss or []
        self.posit_dst = posit_dst or []
        self.negat_dst = negat_dst or []
        self.asetid = asetid or []
        self.nsetid = nsetid or []

    def __iter__(self): return self.__dict__.values().__iter__()

    def copy(self): return HistCollection(self.loss[::], self.posit_dst[::], self.negat_dst[::], self.asetid[::], self.nsetid[::])