import os
import numpy as np

from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import urlparse

import shutil
import zipfile
import rarfile

from typing import Tuple, Literal, Optional
from scipy import io


#%% Archive files URLs

samson_data_URL = "https://web.archive.org/web/20240224052358/https://rslab.ut.ac.ir/documents/81960329/82034930/Data_Matlab.rar"
samson_GT_URL = "https://web.archive.org/web/20240224052358/https://rslab.ut.ac.ir/documents/81960329/82034930/GroundTruth.zip"

jasper_ridge_data_URL = "https://web.archive.org/web/20240224052358/https://rslab.ut.ac.ir/documents/81960329/82034928/jasperRidge2_R198.mat"
jasper_ridge_GT_URL = "https://web.archive.org/web/20240224052358/https://rslab.ut.ac.ir/documents/81960329/82034928/GroundTruth.zip"

urban_data_URL = "https://web.archive.org/web/20240224052358/https://rslab.ut.ac.ir/documents/81960329/82034926/Urban_F210.mat"
urban_6_GT_URL = "https://web.archive.org/web/20240224052358/https://rslab.ut.ac.ir/documents/81960329/82034926/groundTruth_Urban_end6.zip"


#%% Archive download functions

# Convert usual Wayback Machine URL into raw archive-content URL
def wayback_raw_url(url:str) -> str:
    """
    If the URL is a Wayback URL of type /web/<timestamp>/..., convert to 
    /web/<timestamp>id_/... in order to retrieve its raw archived content. 
    Otherwise, return the original URL.
    """
    marker = "/web/"
    if "web.archive.org" not in url or marker not in url:
        return url

    prefix, rest = url.split(marker, 1)
    parts = rest.split("/", 1)
    if len(parts) != 2:
        return url

    timestamp, archived_url = parts
    if not timestamp.endswith("id_"):
        timestamp = timestamp + "id_"

    return f"{prefix}{marker}{timestamp}/{archived_url}"

# Extract ZIP file
def safe_extract_zip(zip_path:Path, target_path:Path) -> None:
    target_path = target_path.resolve()

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = (target_path / member.filename).resolve()
            
            if not str(member_path).startswith(str(target_path)):
                raise ValueError(f"Unsafe ZIP file detected: {member.filename}")

        zf.extractall(target_path)

# Extract RAR file
def safe_extract_rar(rar_path:Path, target_path:Path) -> None:
    target_path = target_path.resolve()

    with rarfile.RarFile(rar_path) as rf:
        for member in rf.infolist():
            member_path = (target_path / member.filename).resolve()
            
            if not str(member_path).startswith(str(target_path)):
                raise ValueError(f"Unsafe RAR archive detected: {member.filename}")

        rf.extractall(target_path)

# Download and extract ZIP file
def download_and_extract_zip(url:str, target_dir:str|Path, name:Optional[str]=None) -> Path:
    """
    Download a ZIP file at 'url', save it in 'target_dir' as 'name', and unzip it in the same directory. 
    Retourn path to the downloaded file.
    """
    target_path = Path(target_dir).expanduser().resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    download_url = wayback_raw_url(url)

    if name is None:
        url_name = Path(urlparse(download_url).path).name
        name = url_name or "downloaded.zip"
    if not name.endswith(".zip"):
        name += ".zip"

    path = target_path / name

    # HTTP request with explicit User-Agent
    request = Request(download_url, headers={"User-Agent": "Python downloader"})

    # Downloading file
    with urlopen(request) as response, open(path, "wb") as f:
        shutil.copyfileobj(response, f)

    # Verifying it is a ZIP file
    if not zipfile.is_zipfile(path):
        raise ValueError(f"The downloaded file is not a ZIP file: {path}")

    # Unzipping file
    safe_extract_zip(path, target_path)

    return path

# Download and extract RAR file
def download_and_extract_rar(url:str, target_dir:str|Path, name:Optional[str]=None) -> Path:
    """
    Download a RAR file at 'url', save it in 'target_dir' as 'name', and extract it in the same directory. 
    Retourn path to the downloaded file.
    """
    target_path = Path(target_dir).expanduser().resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    download_url = wayback_raw_url(url)

    if name is None:
        url_name = Path(urlparse(download_url).path).name
        name = url_name or "downloaded.rar"
    if not name.endswith(".rar"):
        name += ".rar"

    path = target_path / name

    # HTTP request with explicit User-Agent
    request = Request(download_url, headers={"User-Agent": "Python downloader"})

    # Downloading file
    with urlopen(request) as response, open(path, "wb") as f:
        shutil.copyfileobj(response, f)

    # Verifying it is a RAR file
    if not rarfile.is_rarfile(path):
        raise ValueError(f"The downloaded file is not a RAR file: {path}")

    # Extracting file
    safe_extract_rar(path, target_path)

    return path

# Download MAT file
def download_mat_file(url:str, target_dir:str|Path, name:Optional[str]=None) -> Path:
    """
    Download a MAT file at 'url' and save it in 'target_dir' as 'name'. 
    Retourn path to the downloaded file.
    """
    target_dir = os.path.join(target_dir, "Data_Matlab")

    target_path = Path(target_dir).expanduser().resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    download_url = wayback_raw_url(url)

    if name is None:
        url_name = Path(urlparse(download_url).path).name
        name = url_name or "downloaded.mat"
    if not name.endswith(".mat"):
        name += ".mat"

    path = target_path / name

    # HTTP request with explicit User-Agent
    request = Request(download_url, headers={"User-Agent": "Python downloader"})

    # Downloading file
    with urlopen(request) as response, open(path, "wb") as f:
        shutil.copyfileobj(response, f)

    # Verifying it is a MAT file
    try: io.loadmat(path)
    except: raise ValueError(f"The downloaded file is not a loadable MAT file: {path}")

    return path

# Main function to download and extract compressed file
def download_file(url:str, target_dir:str|Path, name:str|None=None) -> Path:
    """
    Download a file at 'url', save it in 'target_dir' as 'name', and --if compressed-- extract it in the same directory. 
    Accepted file formats: {'.zip', '.rar', '.mat'}. 
    Return path to the downloaded file.
    """
    download_url = wayback_raw_url(url)
    url_name = Path(urlparse(download_url).path).name
    
    if url_name.endswith(".zip"):
        return download_and_extract_zip(url, target_dir, name)
    elif url_name.endswith(".rar"):
        return download_and_extract_rar(url, target_dir, name)
    elif url_name.endswith(".mat"):
        return download_mat_file(url, target_dir, name)
    else:
        raise Exception("Given file format is not accepted or must be explicit")


#%% Dataset loader

# Samson dataset
def load_samson(data_path:Optional[str]=None, gt_path:Optional[str]=None, download:bool=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (data, GT_endmembers, A_gtbundances)
    """
    if data_path is None:
        data_path = "./datasets/Samson/Data_Matlab/samson_1.mat"
    if gt_path is None:
        gt_path = "./datasets/Samson/GroundTruth/end3.mat"

    # Download MatLab data cube if necessary
    if not os.path.isfile(data_path):
        if download:
            target_dir = os.path.dirname(os.path.dirname(data_path))
            print("No hyperspectral data found for Samson dataset. Downloading archive file...")
            file_path = download_file(url=samson_data_URL, target_dir=target_dir)
            print(f"Dataset downloaded at:\n{file_path}")
            data_path = os.path.join(target_dir, "Data_Matlab/samson_1.mat")
        else:
            raise Exception("No hyperspectral data found for Samson at the given path. Please, set 'download' to True to unable archive downloading.")

    # Load MatLab data cube
    try: img_obj = io.loadmat(data_path)
    except: raise Exception(f"File {data_path} cannot be loaded as a Matlab file")
    if 'V' in img_obj.keys():
        try: img = img_obj['V']
        except: raise Exception(f"'V' is not a working key for Matlab file {data_path}")
    elif 'Y' in img_obj.keys():
        try: img = img_obj['Y']
        except: raise Exception(f"'Y' is not a working key for Matlab file {data_path}")
    else:
        raise Exception(f"Matlab file {data_path} does not have 'V' nor 'Y' as key")
    if type(img) is not np.ndarray: 
        try: img = np.asarray(img)
        except: raise Exception("Object of key 'V' or 'Y' cannot be converted into a Numpy array")
    
    img = img.astype(np.float32)
    datacube = img.T.reshape(*(int(np.round(np.sqrt(img.shape[-1]))),)*2, img.shape[0]).swapaxes(0,1)
    
    # Download MatLab ground-truth if necessary
    if not os.path.isfile(gt_path):
        if download:
            target_dir = os.path.dirname(os.path.dirname(gt_path))
            print("No ground-truth file found for Samson dataset. Downloading archive file...")
            file_path = download_file(url=samson_GT_URL, target_dir=target_dir)
            print(f"File downloaded at:\n{file_path}")
            gt_path = os.path.join(target_dir, "GroundTruth/end3.mat")
        else:
            raise Exception("No ground-truth file found for Samson at the given path. Please, set 'download' to True to unable archive downloading.")

    # Load ground-truth abundances and endmembers
    gt_data = io.loadmat(gt_path)
    endmembers = np.asarray(gt_data['M']).T
    abundances = np.asarray(gt_data['A']).T.reshape(*datacube.shape[:2], len(endmembers)).swapaxes(0,1)

    # Return triplet (data, GT_endmembers, A_gtbundances)
    return datacube, endmembers, abundances

# Jasper Ridge dataset
def load_jasper_ridge(data_path:Optional[str]=None, gt_path:Optional[str]=None, download:bool=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (data, GT_endmembers, A_gtbundances)
    """
    if data_path is None:
        data_path = "./datasets/JasperRidge/Data_Matlab/jasperRidge2_R198.mat"
    if gt_path is None:
        gt_path = "./datasets/JasperRidge/GroundTruth/end4.mat"

    # Download MatLab data cube if necessary
    if not os.path.isfile(data_path):
        if download:
            target_dir = os.path.dirname(os.path.dirname(data_path))
            print("No hyperspectral data found for Jasper Ridge dataset. Downloading archive file...")
            file_path = download_file(url=jasper_ridge_data_URL, target_dir=target_dir)
            print(f"Dataset downloaded at:\n{file_path}")
            data_path = os.path.join(target_dir, "Data_Matlab/jasperRidge2_R198.mat")
        else:
            raise Exception("No hyperspectral data found for Jasper Ridge at the given path. Please, set 'download' to True to unable archive downloading.")

    # Load MatLab data cube
    try: img_obj = io.loadmat(data_path)
    except: raise Exception(f"File {data_path} cannot be loaded as a Matlab file")
    if 'V' in img_obj.keys():
        try: img = img_obj['V']
        except: raise Exception(f"'V' is not a working key for Matlab file {data_path}")
    elif 'Y' in img_obj.keys():
        try: img = img_obj['Y']
        except: raise Exception(f"'Y' is not a working key for Matlab file {data_path}")
    else:
        raise Exception(f"Matlab file {data_path} does not have 'V' nor 'Y' as key")
    if type(img) is not np.ndarray: 
        try: img = np.asarray(img)
        except: raise Exception("Object of key 'V' or 'Y' cannot be converted into a Numpy array")
    
    img = img.astype(np.float32)
    datacube = img.T.reshape(*(int(np.round(np.sqrt(img.shape[-1]))),)*2, img.shape[0]).swapaxes(0,1)

    # Download MatLab ground-truth if necessary
    if not os.path.isfile(gt_path):
        if download:
            target_dir = os.path.dirname(os.path.dirname(gt_path))
            print("No ground-truth file found for Jasper Ridge dataset. Downloading archive file...")
            file_path = download_file(url=jasper_ridge_GT_URL, target_dir=target_dir)
            print(f"File downloaded at:\n{file_path}")
            gt_path = os.path.join(target_dir, "GroundTruth/end4.mat")
        else:
            raise Exception("No ground-truth file found for Jasper Ridge at the given path. Please, set 'download' to True to unable archive downloading.")

    # Load ground-truth abundances and endmembers
    gt_data = io.loadmat(gt_path)
    endmembers = np.asarray(gt_data['M']).T
    abundances = np.asarray(gt_data['A']).T.reshape(*datacube.shape[:2], len(endmembers)).swapaxes(0,1)

    # Re-arrange labels on abundances and endmembers
    endmembers = np.asarray([endmembers[2], endmembers[0], endmembers[1], endmembers[3]])
    abundances = np.asarray([abundances[...,2], abundances[...,0], abundances[...,1], abundances[...,3]]).transpose(1,2,0)

    # Return triplet (data, GT_endmembers, A_gtbundances)
    return datacube, endmembers, abundances

# Urban-6 dataset
def load_urban6(data_path:Optional[str]=None, gt_path:Optional[str]=None, download:bool=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (data, GT_endmembers, A_gtbundances)
    """
    if data_path is None:
        data_path = "./datasets/Urban/Data_Matlab/Urban_F210.mat"
    if gt_path is None:
        gt_path = "./datasets/Urban/groundTruth_Urban_end6/end6_groundTruth.mat"

    # Download MatLab data cube if necessary
    if not os.path.isfile(data_path):
        if download:
            target_dir = os.path.dirname(os.path.dirname(data_path))
            print("No hyperspectral data found for Urban dataset. Downloading archive file...")
            file_path = download_file(url=urban_data_URL, target_dir=target_dir)
            print(f"Dataset downloaded at:\n{file_path}")
            data_path = os.path.join(target_dir, "Data_Matlab/Urban_F210.mat")
        else:
            raise Exception("No hyperspectral data found for Urban at the given path. Please, set 'download' to True to unable archive downloading.")

    # Load MatLab data cube
    try: img_obj = io.loadmat(data_path)
    except: raise Exception(f"File {data_path} cannot be loaded as a Matlab file")
    if 'V' in img_obj.keys():
        try: img = img_obj['V']
        except: raise Exception(f"'V' is not a working key for Matlab file {data_path}")
    elif 'Y' in img_obj.keys():
        try: img = img_obj['Y']
        except: raise Exception(f"'Y' is not a working key for Matlab file {data_path}")
    else:
        raise Exception(f"Matlab file {data_path} does not have 'V' nor 'Y' as key")
    if type(img) is not np.ndarray: 
        try: img = np.asarray(img)
        except: raise Exception("Object of key 'V' or 'Y' cannot be converted into a Numpy array")
    
    img = img.astype(np.float32)
    datacube = img.T.reshape(*(int(np.round(np.sqrt(img.shape[-1]))),)*2, img.shape[0]).swapaxes(0,1)

    # Remove 48 noisy bands (from 210 to 162 bands)
    if datacube.shape[-1] == 210:
        bad_1based = list(range(1,5)) + [76, 87] + list(range(101,112)) + list(range(136,154)) + list(range(198,211))

        bad = np.array([i-1 for i in bad_1based])
        all_bands = np.arange(datacube.shape[-1])
        good = np.setdiff1d(all_bands, bad)

        datacube = datacube[:, :, good]
        print("F210:", datacube.shape[:-1]+(210,), "-> R162:", datacube.shape)

    # Download MatLab ground-truth if necessary
    if not os.path.isfile(gt_path):
        if download:
            target_dir = os.path.dirname(os.path.dirname(gt_path))
            print("No ground-truth file found for Urban-6 dataset. Downloading archive file...")
            file_path = download_file(url=urban_6_GT_URL, target_dir=target_dir)
            print(f"File downloaded at:\n{file_path}")
            gt_path = os.path.join(target_dir, "groundTruth_Urban_end6/end6_groundTruth.mat")
        else:
            raise Exception("No ground-truth file found for Urban-6 at the given path. Please, set 'download' to True to unable archive downloading.")

    # Load ground-truth abundances and endmembers
    gt_data = io.loadmat(gt_path)
    endmembers = np.asarray(gt_data['M']).T
    abundances = np.asarray(gt_data['A']).T.reshape(*datacube.shape[:2], len(endmembers)).swapaxes(0,1)

    # Return triplet (data, GT_endmembers, A_gtbundances)
    return datacube, endmembers, abundances

# Data loader
def load_dataset(
        data_name:Literal['samson','jasper-ridge','urban6'], 
        data_path:Optional[str]=None, 
        gt_path:Optional[str]=None, 
        download:bool=True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (data, GT_endmembers, A_gtbundances)
    """
    if 'samson' in data_name.lower():
        return load_samson(data_path, gt_path, download)
    elif 'jasper' in data_name.lower() or 'ridge' in data_name.lower():
        return load_jasper_ridge(data_path, gt_path, download)
    elif 'urban' in data_name.lower() and '6' in data_name.lower():
        return load_urban6(data_path, gt_path, download)
    raise ValueError("Wrong data name.")

