import pickle
from os import makedirs
from os.path import split 

def lambda_list():
    return []

def load_pkl_from_path(path):
    path, filename = split(path)
    pkl = load_pkl(filename, '{}/'.format(path))
    return pkl

def load_pkl(filename, folder=''):
    with open(folder + filename + '.pkl', 'rb') as pkl:
        return pickle.load(pkl)

def save_pkl(filename, obj, folder=''):
    """Save pkl file 
    
    Arguments:
        filename {str} 
        obj {pkl object} -- the pickle object to save
    
    Keyword Arguments:
        folder {str} -- The folder to save the file in; it must be appended with '/' (default: {''})
    
    Returns:
        None
    """
    def write_to_pkl():
        with open(folder + filename + '.pkl', 'wb') as pkl:
            return pickle.dump(obj, pkl)
    if folder != '':
        try:
            makedirs(folder)
        except FileExistsError:
            pass
    write_to_pkl()
    

def get_pkl(obj):
    return pickle.dumps(obj)