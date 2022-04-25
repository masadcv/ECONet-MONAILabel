import logging

import numpy as np
import torch
import numpymaxflow
from monai.networks.layers import GaussianMixtureModel
from monailabel.scribbles.utils import make_histograms
from sklearn.mixture import GaussianMixture


def get_eps(data):
    return np.finfo(data.dtype).eps

def maxflow(image, prob, lamda=5, sigma=0.1):
    # lamda: weight of smoothing term
    # sigma: std of intensity values
    return numpymaxflow.maxflow(image, prob, lamda, sigma)

def sklearn_fit_gmm(image, scrib, scribbles_bg_label, scribbles_fg_label, n_components):
    bg = GaussianMixture(n_components=n_components)
    bg.fit(image[scrib == scribbles_bg_label].reshape(-1, 1))

    fg = GaussianMixture(n_components=n_components)
    fg.fit(image[scrib == scribbles_fg_label].reshape(-1, 1))

    return bg, fg


def learn_and_apply_gmm_sklearn(
    image, scrib, scribbles_bg_label, scribbles_fg_label, mixture_size
):
    # based on https://github.com/jiviteshjain/grabcut/blob/main/src/grabcut.ipynb
    bg_gmm, fg_gmm = sklearn_fit_gmm(
        image, scrib, scribbles_bg_label, scribbles_fg_label, mixture_size
    )

    # add empty channel to image and scrib to be inline with pytorch layout
    bg_prob = (
        bg_gmm.score_samples(image.reshape((-1, 1)))
        .reshape(image.shape)
        .astype(np.float32)
    )
    fg_prob = (
        fg_gmm.score_samples(image.reshape((-1, 1)))
        .reshape(image.shape)
        .astype(np.float32)
    )

    bg_prob = np.exp(bg_prob)
    fg_prob = np.exp(fg_prob)

    gmm_output = np.concatenate([bg_prob, fg_prob])

    return gmm_output


def learn_and_apply_gmm_monai(
    image, scrib, scribbles_bg_label, scribbles_fg_label, mixture_size
):
    # this function is limited to binary segmentation at the moment
    n_classes = 2

    # make trimap
    trimap = np.zeros_like(scrib).astype(np.int32)

    # fetch anything that is not scribbles
    not_scribbles = ~((scrib == scribbles_bg_label) | (scrib == scribbles_fg_label))

    # set these to 0 == unused
    trimap[not_scribbles] = -1

    # set background scrib to -1
    trimap[scrib == scribbles_bg_label] = 0
    # set foreground scrib to 1
    trimap[scrib == scribbles_fg_label] = 1

    # image_tri = image[~not_scribbles]
    # trimap = trimap[~not_scribbles]

    # add empty channel to image and scrib to be inline with pytorch layout
    image = np.expand_dims(image, axis=0)
    # image_tri = np.expand_dims(image_tri, axis=0)
    trimap = np.expand_dims(trimap, axis=0)

    # transfer everything to pytorch tensor,
    # we only use CUDA as GMM from MONAI is only available on CUDA atm (2021/03/12)
    # if no cuda device found, then exit now
    if not torch.cuda.is_available():
        raise IOError("Unable to find CUDA device, check your torch/monai installation")

    device = "cuda"
    image = torch.from_numpy(image).type(torch.float32).to(device)
    trimap = torch.from_numpy(trimap).type(torch.int32).to(device)

    # initialise our GMM
    gmm = GaussianMixtureModel(
        image.size(1),
        mixture_count=n_classes,
        mixture_size=mixture_size,
        verbose_build=False,
    )
    # gmm.reset()

    # learn gmm from image_tri and trimap
    gmm.learn(image, trimap)

    # apply gmm on image
    gmm_output = gmm.apply(image)

    return gmm_output.squeeze(0).cpu().numpy()


def make_likelihood_image_gmm(
    image,
    scrib,
    scribbles_bg_label,
    scribbles_fg_label,
    return_label=False,
    mixture_size=20,
):
    # learn gmm and apply to image, return output label prob
    try:
        # this may fail if MONAI Cpp Extensions are not loaded properly
        retprob = learn_and_apply_gmm_monai(
            image=image,
            scrib=scrib,
            scribbles_bg_label=scribbles_bg_label,
            scribbles_fg_label=scribbles_fg_label,
            mixture_size=mixture_size,
        )
    except:
        logging.info("Unable to run MONAI's GMM, falling back to sklearn GMM")
        retprob = learn_and_apply_gmm_sklearn(
            image=image,
            scrib=scrib,
            scribbles_bg_label=scribbles_bg_label,
            scribbles_fg_label=scribbles_fg_label,
            mixture_size=mixture_size,
        )

    # if needed, convert to discrete labels instead of probability
    if return_label:
        retprob = np.expand_dims(np.argmax(retprob, axis=0), axis=0).astype(np.float32)

    return retprob


def make_likelihood_image_histogram(
    image,
    scrib,
    scribbles_bg_label,
    scribbles_fg_label,
    return_label=False,
    alpha_bg=1,
    alpha_fg=1,
    bins=32,
):
    # normalise image in range [0, 1] if needed
    min_img = np.min(image)
    max_img = np.max(image)
    if min_img < 0.0 or max_img > 1.0:
        image = (image - min_img) / (max_img - min_img)

    # generate histograms for background/foreground
    bg_hist, fg_hist, bin_edges = make_histograms(
        image,
        scrib,
        scribbles_bg_label,
        scribbles_fg_label,
        alpha_bg=alpha_bg,
        alpha_fg=alpha_fg,
        bins=bins,
    )

    # lookup values for each voxel for generating background/foreground probabilities
    dimage = np.digitize(image, bin_edges[:-1]) - 1
    fprob = fg_hist[dimage]
    bprob = bg_hist[dimage]

    retprob = np.concatenate([bprob, fprob], axis=0)

    # if needed, convert to discrete labels instead of probability
    if return_label:
        retprob = np.expand_dims(np.argmax(retprob, axis=0), axis=0).astype(np.float32)

    return retprob
