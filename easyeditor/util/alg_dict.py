from ..models.rome import ROMEHyperParams, apply_rome_to_model
from ..models.memit import MEMITHyperParams, apply_memit_to_model
from ..models.ft import FTHyperParams, apply_ft_to_model
from ..dataset import ZsreDataset, CounterFactDataset
from ..models.grace import GraceHyperParams, apply_grace_to_model
from ..models.wise import WISEHyperParams, apply_wise_to_model
from ..models.emmet import EMMETHyperParams, apply_emmet_to_model
from ..models.alphaedit import AlphaEditHyperParams, apply_AlphaEdit_to_model
from ..models.asmem import ASMemHyperParams, apply_asmem_to_model
from ..models.ndedit import NDEditHyperParams, apply_ndedit_to_model
from ..models.deltaedit import DeltaEditHyperParams, apply_deltaedit_to_model
from ..models.eamet import EAMETHyperParams, apply_eamet_to_model
from ..models.blue import BLUEHyperParams, apply_blue_to_model, apply_prune_to_model, apply_rect_to_model

ALG_DICT = {
    'ROME': apply_rome_to_model,
    'MEMIT': apply_memit_to_model,
    'FT': apply_ft_to_model,
    'GRACE': apply_grace_to_model,
    'WISE': apply_wise_to_model,
    'EMMET': apply_emmet_to_model,
    'AlphaEdit': apply_AlphaEdit_to_model,
    'ASMem': apply_asmem_to_model,
    'ASMem': apply_asmem_to_model,
    'NDEdit': apply_ndedit_to_model,
    'DeltaEdit': apply_deltaedit_to_model,
    'EAMET': apply_eamet_to_model,
    'BLUE': apply_blue_to_model,
    'PRUNE': apply_prune_to_model,
    'RECT': apply_rect_to_model,
}

ALG_MULTIMODAL_DICT = {}

PER_ALG_DICT = {}

DS_DICT = {
    "cf": CounterFactDataset,
    "zsre": ZsreDataset,
}
