import numpy as np
import matplotlib.pyplot as plt

from typing import Literal, Optional

from src.datasets import load_dataset
from src.polyset import normed


#%% Array normalization

def normalized(
        array:np.ndarray, 
        axis:Optional[tuple]=None, 
        output_range:Optional[tuple]=None, 
        output_dtype:Optional[np.dtype]=None
) -> np.ndarray:
    if output_dtype is None:
        if np.issubdtype(array.dtype, np.integer):
            output_dtype = np.float64
        else:
            output_dtype = array.dtype
    if output_range is None:
        if np.issubdtype(output_dtype, np.integer):
            output_range = (np.iinfo(output_dtype).min, np.iinfo(output_dtype).max)
        else:
            output_range = (0., 1.)
    if axis is None:
        axis = tuple(np.arange(array.ndim, dtype=int))
    a_min = array.min(axis, keepdims=True)
    delta = array.max(axis, keepdims=True) - a_min
    diff_zero = delta < np.finfo(array.dtype).resolution
    float_out = (output_range[1] - output_range[0]) * ~diff_zero / (delta * ~diff_zero + diff_zero)
    float_out = (array - a_min) * float_out + output_range[0]
    if np.issubdtype(output_dtype, np.integer):
        return np.round(float_out).astype(output_dtype)
    elif float_out.dtype != output_dtype:
        return float_out.astype(output_dtype)
    return float_out


#%% Dataset displayer

# Show Samson
def show_samson(image:Optional[np.ndarray]=None, M_gt:Optional[np.ndarray]=None, A_gt:Optional[np.ndarray]=None, dpi:Optional[float]=None) -> None:
    """
    Display Samson dataset.
    """
    if image is None or M_gt is None or A_gt is None:
        image_ori, M_gt_ori, A_gt_ori = load_dataset('samson', download=False)
        if image is None: image = image_ori
        if M_gt  is None: M_gt  = M_gt_ori
        if A_gt  is None: A_gt  = A_gt_ori

    fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [80, 80, 100]}, figsize=(20,6))

    ax[0].imshow(normalized(np.transpose((
        normalized(  0.6 * normalized(image[:,:,98]) + 0.2 * normalized(image[:,:,90]) + 0.7 * normalized(image[:,:,50])), 
        normalized(  0.1 * normalized(image[:,:,98]) - 0.3 * normalized(image[:,:,90]) + 0.8 * normalized(image[:,:,50])), 
        normalized(- 0.5 * normalized(image[:,:,98]) + 0.4 * normalized(image[:,:,90]) + 0.9 * normalized(image[:,:,50])), 
    ), axes=(1,2,0))))
    ax[0].set_axis_off()
    ax[0].set_title("Samson (Y)")

    ax[1].imshow(normalized(A_gt[...,:3]))
    ax[1].set_axis_off()
    ax[1].set_title("GT: Aboundances (A)")

    X = np.arange(M_gt.shape[1])
    for i in range(M_gt.shape[0]):
        col = (int(i==0), 0.8*int(i==1), int(i==2))
        lab = "Soil" * int(i==0) + "Tree" * int(i==1) + "Water" * int(i==2)
        ax[2].plot(X, M_gt[i], c=col, label=lab)
    ax[2].legend()
    ax[2].set_xlim(-0.10, 156.5)
    ax[2].set_ylim(-0.01, 1.03 )
    ax[2].set_ylabel("Reflectance")
    ax[2].set_xlabel("Bands")
    ax[2].grid(linestyle="--")
    ax[2].set_aspect(117)
    ax[2].set_title("GT: Endmembers (M)")

    if dpi is not None: fig.set_dpi(dpi)
    plt.show()

# Show Jasper Ridge
def show_jasper_ridge(image:Optional[np.ndarray]=None, M_gt:Optional[np.ndarray]=None, A_gt:Optional[np.ndarray]=None, dpi:Optional[float]=None) -> None:
    """
    Display Jasper Ridge dataset.
    """
    if image is None or M_gt is None or A_gt is None:
        image_ori, M_gt_ori, A_gt_ori = load_dataset('jasper-ridge', download=False)
        if image is None: image = image_ori
        if M_gt  is None: M_gt  = M_gt_ori
        if A_gt  is None: A_gt  = A_gt_ori

    fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [80, 80, 100]}, figsize=(20,6))

    ax[0].imshow(normalized(np.transpose((
        normalized(  1.0 * normalized(image[:,:,150]) + 0.0 * normalized(image[:,:,73]) - 0.0 * normalized(image[:,:,15])), 
        normalized(  1.0 * normalized(image[:,:,150]) + 1.0 * normalized(image[:,:,73]) + 3.0 * normalized(image[:,:,15])), 
        normalized(- 2.0 * normalized(image[:,:,150]) - 0.2 * normalized(image[:,:,73]) + 8.0 * normalized(image[:,:,15]))
    ), axes=(1,2,0))))
    ax[0].set_axis_off()
    ax[0].set_title("Jasper Ridge (Y)")

    ax[1].imshow(normalized(A_gt[...,:3]))
    ax[1].set_axis_off()
    ax[1].set_title("GT: Aboundances (A)")

    X = np.arange(M_gt.shape[1])
    for i in range(M_gt.shape[0]):
        col = (int(i==0), 0.8*int(i==1), int(i==2))
        lab = "Soil" * int(i==0) + "Tree" * int(i==1) + "Water" * int(i==2) + "Road" * int(i>2)
        ax[2].plot(X, M_gt[i], c=col, label=lab)
    ax[2].legend()
    ax[2].set_xlim(-0.10, 198.5)
    ax[2].set_ylim(-0.01, 0.65 )
    ax[2].set_ylabel("Reflectance")
    ax[2].set_xlabel("Bands")
    ax[2].grid(linestyle="--")
    ax[2].set_aspect(235)
    ax[2].set_title("GT: Endmembers (M)")

    if dpi is not None: fig.set_dpi(dpi)
    plt.show()

# Show Urban-6
def show_urban6(image:Optional[np.ndarray]=None, M_gt:Optional[np.ndarray]=None, A_gt:Optional[np.ndarray]=None, dpi:Optional[float]=None) -> None:
    """
    Display Urban-6 dataset.
    """
    if image is None or M_gt is None or A_gt is None:
        image_ori, M_gt_ori, A_gt_ori = load_dataset('urban6', download=False)
        if image is None: image = image_ori
        if M_gt  is None: M_gt  = M_gt_ori
        if A_gt  is None: A_gt  = A_gt_ori

    fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [80, 80, 100]}, figsize=(20,6))

    ax[0].imshow(normalized(np.transpose((
        normalized(  0.6 * normalized(image[:,:,98]) + 0.2 * normalized(image[:,:,90]) + 0.7 * normalized(image[:,:,50])), 
        normalized(  0.1 * normalized(image[:,:,98]) + 0.9 * normalized(image[:,:,90]) + 0.1 * normalized(image[:,:,50])), 
        normalized(- 0.9 * normalized(image[:,:,98]) + 0.4 * normalized(image[:,:,90]) + 0.9 * normalized(image[:,:,50]))
    ), axes=(1,2,0))))
    ax[0].set_axis_off()
    ax[0].set_title("Urban (Y)")

    img_label = np.zeros(A_gt.shape[:-1]+(3,), dtype=np.uint8)
    for i in range(A_gt.shape[-1]):
        if i in {0,3,4}: img_label[...,0]+= (normalized(A_gt)[...,i] * 255).astype(np.uint8)
        if i in {1,3,5}: img_label[...,1]+= (normalized(A_gt)[...,i] * 255).astype(np.uint8)
        if i in {2,4,5}: img_label[...,2]+= (normalized(A_gt)[...,i] * 255).astype(np.uint8)
    ax[1].imshow(img_label)
    ax[1].set_axis_off()
    ax[1].set_title("GT: Aboundances (A)")

    X = np.arange(M_gt.shape[1])
    for i in range(M_gt.shape[0]):
        col = (int(i in {0,3,4}), 0.8*int(i in {1,3,5}), int(i in {2,4,5}))
        lab = "Asphalt" * int(i==0) + "Grass" * int(i==1) + "Tree" * int(i==2) + "Roof" * int(i==3) + "Metal" * int(i==4) + "Dirt" * int(i==5)
        ax[2].plot(X, M_gt[i], c=col, label=lab)
    ax[2].legend()
    ax[2].set_xlim(-0.10, 162.5)
    ax[2].set_ylim(-0.01, 0.602)
    ax[2].set_ylabel("Reflectance")
    ax[2].set_xlabel("Bands")
    ax[2].grid(linestyle="--")
    ax[2].set_aspect(205)
    ax[2].set_title("GT: Endmembers (M)")

    if dpi is not None: fig.set_dpi(dpi)
    plt.show()

# Data displayer
def show_dataset(
        data_name:Literal['samson','jasper-ridge','urban6'], 
        image:Optional[np.ndarray] = None, 
        M_gt :Optional[np.ndarray] = None, 
        A_gt :Optional[np.ndarray] = None, 
        dpi:Optional[float] = None
) -> None:
    """
    Display dataset.
    """
    if 'samson' in data_name.lower():
        return show_samson(image, M_gt, A_gt, dpi)
    elif 'jasper' in data_name.lower() or 'ridge' in data_name.lower():
        return show_jasper_ridge(image, M_gt, A_gt, dpi)
    elif 'urban' in data_name.lower() and '6' in data_name.lower():
        return show_urban6(image, M_gt, A_gt, dpi)
    raise ValueError("Wrong data name.")


#%% Show endmember results

# Show endmember results on Samson
def show_endmember_results_samson(M_gt:np.ndarray, M_hat:np.ndarray) -> None:
    X = np.arange(M_gt.shape[-1])
    plt.plot(X, normed(M_gt [0]), c=(1.0, 0.5, 0.5), label='Soil')
    plt.plot(X, normed(M_hat[0]), c=(0.4, 0.1, 0.1))
    plt.plot(X, normed(M_gt [1]), c=(0.5, 1.0, 0.5), label='Tree')
    plt.plot(X, normed(M_hat[1]), c=(0.1, 0.4, 0.1))
    plt.plot(X, normed(M_gt [2]), c=(0.5, 0.5, 1.0), label='Water')
    plt.plot(X, normed(M_hat[2]), c=(0.1, 0.1, 0.4))
    plt.legend()
    plt.xlim(-0.10, 156.5)
    plt.ylim(-0.01, 0.158)
    plt.ylabel("Reflectance")
    plt.xlabel("Bands")
    plt.grid(linestyle="--")
    plt.title("Light: GT   |   Dark: estimated")
    plt.show()

# Show endmember results on Jasper Ridge
def show_endmember_results_jasper_ridge(M_gt:np.ndarray, M_hat:np.ndarray) -> None:
    X = np.arange(M_gt.shape[-1])
    plt.plot(X, normed(M_gt [0]), c=(1.0, 0.5, 0.5), label='Soil')
    plt.plot(X, normed(M_hat[0]), c=(0.4, 0.1, 0.1))
    plt.plot(X, normed(M_gt [1]), c=(0.5, 1.0, 0.5), label='Tree')
    plt.plot(X, normed(M_hat[1]), c=(0.1, 0.4, 0.1))
    plt.plot(X, normed(M_gt [2]), c=(0.5, 0.5, 1.0), label='Water')
    plt.plot(X, normed(M_hat[2]), c=(0.1, 0.1, 0.4))
    plt.plot(X, normed(M_gt [3]), c=(0.4, 0.4, 0.4), label='Road')
    plt.plot(X, normed(M_hat[3]), c=(0.0, 0.0, 0.0))
    plt.legend()
    plt.xlim(-0.10, 198.5)
    plt.ylim(-0.01, 0.230)
    plt.ylabel("Reflectance")
    plt.xlabel("Bands")
    plt.grid(linestyle="--")
    plt.title("Light: GT   |   Dark: estimated")
    plt.show()

# Show endmember results on Urban-6
def show_endmember_results_urban6(M_gt:np.ndarray, M_hat:np.ndarray) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    for i in range(2): # two plots (3 endmembers per plot)
        X = np.arange(M_gt.shape[-1])
        k = i * 3 + 0; col = (1.0 * int(k in {0,3,4}), 0.8 * int(k in {1,3,5}), 1.0 * int(k in {2,4,5}))
        ax[i].plot(X, normed(M_gt [3*i+0]), c=col, label='Asphalt'*(i==0)+'Roof' *(i==1))
        k = i * 3 + 0; col = (0.6 * int(k in {0,3,4}), 0.5 * int(k in {1,3,5}), 0.6 * int(k in {2,4,5}))
        ax[i].plot(X, normed(M_hat[3*i+0]), c=col)
        k = i * 3 + 1; col = (1.0 * int(k in {0,3,4}), 0.8 * int(k in {1,3,5}), 1.0 * int(k in {2,4,5}))
        ax[i].plot(X, normed(M_gt [3*i+1]), c=col, label='Grass'  *(i==0)+'Metal'*(i==1))
        k = i * 3 + 1; col = (0.6 * int(k in {0,3,4}), 0.5 * int(k in {1,3,5}), 0.6 * int(k in {2,4,5}))
        ax[i].plot(X, normed(M_hat[3*i+1]), c=col)
        k = i * 3 + 2; col = (1.0 * int(k in {0,3,4}), 0.8 * int(k in {1,3,5}), 1.0 * int(k in {2,4,5}))
        ax[i].plot(X, normed(M_gt [3*i+2]), c=col, label='Tree'   *(i==0)+'Dirt' *(i==1))
        k = i * 3 + 2; col = (0.6 * int(k in {0,3,4}), 0.5 * int(k in {1,3,5}), 0.6 * int(k in {2,4,5}))
        ax[i].plot(X, normed(M_hat[3*i+2]), c=col)
        ax[i].legend()
        ax[i].set_xlim(-0.10, 162.5)
        ax[i].set_ylim(-0.01, 0.175)
        ax[i].set_ylabel("Reflectance")
        ax[i].set_xlabel("Bands")
        ax[i].grid(linestyle="--")
        ax[i].set_title("Light: GT   |   Dark: estimated")
    plt.show()

# Endmember results displayer
def show_endmember_results(
        data_name:Literal['samson','jasper-ridge','urban6'], 
        M_gt:np.ndarray, 
        M_hat:np.ndarray
) -> None:
    """
    Display dataset.
    """
    if 'samson' in data_name.lower():
        return show_endmember_results_samson(M_gt, M_hat)
    elif 'jasper' in data_name.lower() or 'ridge' in data_name.lower():
        return show_endmember_results_jasper_ridge(M_gt, M_hat)
    elif 'urban' in data_name.lower() and '6' in data_name.lower():
        return show_endmember_results_urban6(M_gt, M_hat)
    raise ValueError("Wrong data name.")


#%% Show abundance results

# Show abundance results on Samson
def show_abundance_results_samson(A_gt:np.ndarray, A_hat:np.ndarray, cmap:str="magma", dpi:Optional[float]=None) -> None:
    fig, ax = plt.subplots(1,A_gt.shape[-1]+1,figsize=(4.5*(A_gt.shape[-1]+1),4))
    disp_img = A_gt[:,:,:3]; disp_img[disp_img <= 0] = 0; disp_img[disp_img >= 1] = 1
    ax[0].imshow(disp_img)
    ax[0].set_axis_off()
    for i in range(A_gt.shape[-1]):
        disp_img = A_gt[:,:,i]; disp_img[disp_img <= 0] = 0; disp_img[disp_img >= 1] = 1
        ax[i+1].imshow(disp_img, cmap=cmap)
        ax[i+1].set_axis_off()
    fig.suptitle(f"GT abundances", fontsize="x-large")
    if dpi is not None: fig.set_dpi(dpi)
    plt.show()

    fig, ax = plt.subplots(1,A_hat.shape[-1]+1,figsize=(4.5*(A_hat.shape[-1]+1),4))
    disp_img = A_hat[:,:,:3]; disp_img[disp_img <= 0] = 0; disp_img[disp_img >= 1] = 1
    ax[0].imshow(disp_img)
    ax[0].set_axis_off()
    for i in range(A_hat.shape[-1]):
        disp_img = A_hat[:,:,i]; disp_img[disp_img <= 0] = 0; disp_img[disp_img >= 1] = 1
        ax[i+1].imshow(disp_img, cmap=cmap)
        ax[i+1].set_axis_off()
    fig.suptitle(f"Estimated abundances", fontsize="x-large")
    if dpi is not None: fig.set_dpi(dpi)
    plt.show()

# Show abundance results on Jasper Ridge
def show_abundance_results_jasper_ridge(A_gt:np.ndarray, A_hat:np.ndarray, cmap:str="magma", dpi:Optional[float]=None) -> None:
    fig, ax = plt.subplots(1,A_gt.shape[-1]+1,figsize=(4.5*(A_gt.shape[-1]+1),4))
    disp_img = A_gt[:,:,:3]; disp_img[disp_img <= 0] = 0; disp_img[disp_img >= 1] = 1
    ax[0].imshow(disp_img)
    ax[0].set_axis_off()
    for i in range(A_gt.shape[-1]):
        disp_img = A_gt[:,:,i]; disp_img[disp_img <= 0] = 0; disp_img[disp_img >= 1] = 1
        ax[i+1].imshow(disp_img, cmap=cmap)
        ax[i+1].set_axis_off()
    fig.suptitle(f"GT abundances", fontsize="x-large")
    if dpi is not None: fig.set_dpi(dpi)
    plt.show()

    fig, ax = plt.subplots(1,A_hat.shape[-1]+1,figsize=(4.5*(A_hat.shape[-1]+1),4))
    disp_img = A_hat[:,:,:3]; disp_img[disp_img <= 0] = 0; disp_img[disp_img >= 1] = 1
    ax[0].imshow(disp_img)
    ax[0].set_axis_off()
    for i in range(A_hat.shape[-1]):
        disp_img = A_hat[:,:,i]; disp_img[disp_img <= 0] = 0; disp_img[disp_img >= 1] = 1
        ax[i+1].imshow(disp_img, cmap=cmap)
        ax[i+1].set_axis_off()
    fig.suptitle(f"Estimated abundances", fontsize="x-large")
    if dpi is not None: fig.set_dpi(dpi)
    plt.show()

# Show abundance results on Urban-6
def show_abundance_results_urban6(A_gt:np.ndarray, A_hat:np.ndarray, cmap:str="magma", dpi:Optional[float]=None) -> None:
    fig, ax = plt.subplots(1,A_gt.shape[-1]+1,figsize=(4.5*(A_gt.shape[-1]+1),4))
    img_label = np.zeros(A_gt.shape[:-1]+(3,), dtype=np.uint8)
    for i in range(A_gt.shape[-1]):
        disp_img = A_gt[...,i]; disp_img[disp_img <= 0] = 0; disp_img[disp_img >= 1] = 1
        if i in {0,3,4}: img_label[...,0] += (disp_img * 255).astype(np.uint8)
        if i in {1,3,5}: img_label[...,1] += (disp_img * 255).astype(np.uint8)
        if i in {2,4,5}: img_label[...,2] += (disp_img * 255).astype(np.uint8)
    ax[0].imshow(img_label)
    ax[0].set_axis_off()
    for i in range(A_gt.shape[-1]):
        ax[i+1].imshow(A_gt[:,:,i], cmap=cmap)
        ax[i+1].set_axis_off()
    fig.suptitle(f"GT abundances", fontsize="x-large")
    if dpi is not None: fig.set_dpi(dpi)
    plt.show()

    fig, ax = plt.subplots(1,A_hat.shape[-1]+1,figsize=(4.5*(A_hat.shape[-1]+1),4))
    img_label = np.zeros(A_hat.shape[:-1]+(3,), dtype=np.uint8)
    for i in range(A_hat.shape[-1]):
        disp_img = A_hat[...,i]; disp_img[disp_img <= 0] = 0; disp_img[disp_img >= 1] = 1
        if i in {0,3,4}: img_label[...,0] += (disp_img * 255).astype(np.uint8)
        if i in {1,3,5}: img_label[...,1] += (disp_img * 255).astype(np.uint8)
        if i in {2,4,5}: img_label[...,2] += (disp_img * 255).astype(np.uint8)
    ax[0].imshow(img_label)
    ax[0].set_axis_off()
    for i in range(A_hat.shape[-1]):
        ax[i+1].imshow(A_hat[:,:,i], cmap=cmap)
        ax[i+1].set_axis_off()
    fig.suptitle(f"Estimated abundances", fontsize="x-large")
    if dpi is not None: fig.set_dpi(dpi)
    plt.show()

# Abundance results displayer
def show_abundance_results(
        data_name:Literal['samson','jasper-ridge','urban6'], 
        A_gt:np.ndarray, 
        A_hat:np.ndarray, 
        cmap:str = "magma", 
        dpi:Optional[float] = None
) -> None:
    """
    Display dataset.
    """
    if 'samson' in data_name.lower():
        return show_abundance_results_samson(A_gt, A_hat, cmap, dpi)
    elif 'jasper' in data_name.lower() or 'ridge' in data_name.lower():
        return show_abundance_results_jasper_ridge(A_gt, A_hat, cmap, dpi)
    elif 'urban' in data_name.lower() and '6' in data_name.lower():
        return show_abundance_results_urban6(A_gt, A_hat, cmap, dpi)
    raise ValueError("Wrong data name.")

