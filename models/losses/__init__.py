import copy

# basic_loss
# from .basic_loss import LossFromOutput

# det loss
from .det_db_loss import DBLoss
# from .det_east_loss import EASTLoss
# from .det_sast_loss import SASTLoss
# from .det_pse_loss import PSELoss
# from .det_fce_loss import FCELoss

# rec loss
from .rec_ctc_loss import CTCLoss
# from .rec_att_loss import AttentionLoss
# from .rec_srn_loss import SRNLoss
# from .rec_nrtr_loss import NRTRLoss
from .rec_sar_loss import SARLoss
# from .rec_aster_loss import AsterLoss
# from .rec_pren_loss import PRENLoss
from .rec_multi_loss import MultiLoss
from .multibox_loss import MultiBoxLoss
# cls loss
# from .cls_loss import ClsLoss

# e2e loss
# from .e2e_pg_loss import PGLoss
# from .kie_sdmgr_loss import SDMGRLoss

# basic loss function
# from .basic_loss import DistanceLoss

# combined loss function
from .combined_loss import CombinedLoss

# table loss
# from .table_att_loss import TableAttentionLoss

# vqa token loss
# from .vqa_token_layoutlm_loss import VQASerTokenLayoutLMLoss
from .lang_loss import LangMultiLosses

def build_loss(config):
    support_dict = [
        'DBLoss', 'PSELoss', 'EASTLoss', 'SASTLoss', 'FCELoss', 'CTCLoss',
        'ClsLoss', 'AttentionLoss', 'SRNLoss', 'PGLoss', 'CombinedLoss',
        'NRTRLoss', 'TableAttentionLoss', 'SARLoss', 'AsterLoss', 'SDMGRLoss',
        'VQASerTokenLayoutLMLoss', 'LossFromOutput', 'PRENLoss', 'MultiLoss', 'MultiBoxLoss'
    ]
    config = copy.deepcopy(config)
    module_name = config.pop('name')
    assert module_name in support_dict, Exception('loss only support {}'.format(
        support_dict))
    module_class = eval(module_name)(**config)
    return module_class
