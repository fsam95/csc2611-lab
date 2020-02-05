import pandas as pd
from pkl_io import load_pkl
def _load_synonymy():
    synonymy_df = pd.read_csv('synonymy.csv')


def get_rg_brown_sim_df():
    rg_and_brown_sim_df = load_pkl('rg_sim_values')
    return rg_and_brown_sim_df