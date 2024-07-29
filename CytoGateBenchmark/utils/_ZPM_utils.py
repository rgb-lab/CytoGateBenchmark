import FACSPy as fp
from anndata import AnnData

def _create_dataset(input_directory: str) -> AnnData:
    return fp.read_dataset(input_dir = input_directory, file_name = "raw_data")
