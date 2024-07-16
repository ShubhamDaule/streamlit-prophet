from typing import Any, Dict, Tuple
import io
from pathlib import Path
import pandas as pd
# import requests
import streamlit as st
import toml
from PIL import Image

def get_project_root() -> str:
    """Returns project root path.
    Returns
    -------
    str
        Project root path.
    """
    return str(Path(__file__).parent.parent.parent)

@st.cache_data
def load_toy_dataset(url: str) -> pd.DataFrame:
    """ a toy dataset from an external source and converts it into a pandas dataframe.
    Parameters
    ----------
    url : str
        Link to the toy dataset.
    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    # download = requests.get(url).content
    df = pd.read_excel(url)
    return df


@st.cache_data
def load_custom_config(config_file: io.BytesIO) -> Dict[Any, Any]:
    """Loads config toml file from user's file system as a dictionary.
    Parameters
    ----------
    config_file
        Uploaded toml config file.
    Returns
    -------
    dict
        Loaded config dictionary.
    """
    toml_file = Path(get_project_root()) / f"config/custom_{config_file.name}"
    write_bytesio_to_file(str(toml_file), config_file)
    config = toml.load(toml_file)
    return dict(config)

def write_bytesio_to_file(filename: str, bytesio: io.BytesIO) -> None:
    """
    Write the contents of the given BytesIO to a file.
    Parameters
    ----------
    filename
        Uploaded toml config file.
    bytesio
        BytesIO object.
    """
    with open(filename, "wb") as outfile:
        outfile.write(bytesio.getbuffer())

  @st.cache_data
def load_image(image_name: str) -> Image:
    """Displays an image.
    Parameters
    ----------
    image_name : str
        Local path of the image.
    Returns
    -------
    Image
        Image to be displayed.
    """
    return Image.open(Path(get_project_root()) / f"streamlit2/utils/{image_name}")


@st.cache_data
def load_dataset(file: str) -> pd.DataFrame:  #, load_options: Dict[Any, Any]
    """Loads dataset from user's file system as a pandas dataframe.
    Parameters
    ----------
    file
        Uploaded dataset file.
    load_options : Dict
        Dictionary containing separator information.
    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    try:
        return pd.read_excel(file) #, sep=load_options["separator"]
    except:
        st.error(
            "This file can't be converted into a dataframe. Please import a excel file with a valid separator."
        )
        st.stop()
