from enum import Enum


class BatchEditor(Enum):
    MEMIT = 'MEMIT'
    FT = 'FT'
    EMMET = "EMMET"
    ALPHAEDIT = "AlphaEdit"
    EAMET = "EAMET"
    BLUE = "BLUE"
    PRUNE = "PRUNE"
    RECT = "RECT"

    @staticmethod
    def is_batchable_method(alg_name: str):
        return alg_name in {
            BatchEditor.MEMIT.value,
            BatchEditor.FT.value,
            BatchEditor.EMMET.value,
            BatchEditor.ALPHAEDIT.value,
            BatchEditor.EAMET.value,
            BatchEditor.BLUE.value,
            BatchEditor.PRUNE.value,
            BatchEditor.RECT.value,
        }
