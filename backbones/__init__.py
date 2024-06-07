from .SubNets.FeatureNets import BERTEncoder
from .FusionNets.MAG_BERT import MAG_BERT
from .FusionNets.MISA import MISA
from .FusionNets.MULT import MULT
from .FusionNets.NMFIR import NMFIR
from .FusionNets.BASELINE import BASELINE

text_backbones_map = {
                    'bert-base-uncased': BERTEncoder
                }

methods_map = {
    'mag_bert': MAG_BERT,
    'misa': MISA,
    'mult': MULT,
    'nmfir' : NMFIR,
    'baseline': BASELINE
}