from collections import namedtuple

_HistCollecion = namedtuple('_HistCollecion', 'loss posit_dst negat_dst asetid nsetid')

class HistCollection(_HistCollecion):
    def __init__(self, loss=[], posit_dst=[], negat_dst=[], asetid=[], nsetid=[]):
        super().__init__(loss, posit_dst, negat_dst, asetid, nsetid)