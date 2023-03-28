import copy
import logging
import math
import os
import pickle
import time
from typing import List

import numpy as np
import torchhaarfeatures
import torch
import tqdm
from monailabel.scribbles.transforms import InteractiveSegmentationTransform
from skimage.util.shape import view_as_windows
from sklearn.ensemble import RandomForestClassifier

from lib.layers import GaussianSmoothing2d, GaussianSmoothing3d, MyDiceCELoss
from lib.utils import (
    get_eps,
    make_likelihood_image_gmm,
    make_likelihood_image_histogram,
    maxflow,
)

from .online_model import ECONetFCNHaarFeatures, ECONetFCNLearnedFeatures

logger = logging.getLogger(__name__)


class MyInteractiveSegmentationTransform(InteractiveSegmentationTransform):
    def __init__(self, meta_key_postfix):
        super().__init__(meta_key_postfix)

    def _copy_affine(self, d, src, dst):
        # make keys
        src_key = "_".join([src, self.meta_key_postfix])
        dst_key = "_".join([dst, self.meta_key_postfix])

        # check if keys exists, if so then copy affine info
        if src_key in d.keys() and "affine" in d[src_key]:
            # create a new destination dictionary if needed
            d[dst_key] = {} if dst_key not in d.keys() else d[dst_key]

            # copy over affine information
            d[dst_key]["affine"] = copy.deepcopy(d[src_key]["affine"])

        if src_key in d.keys() and "pixdim" in d[src_key]:
            # create a new destination dictionary if needed
            d[dst_key] = {} if dst_key not in d.keys() else d[dst_key]

            # copy over affine information
            d[dst_key]["pixdim"] = copy.deepcopy(d[src_key]["pixdim"])

        return d


class MakeLikelihoodFromScribblesECONetd(MyInteractiveSegmentationTransform):
    def __init__(
        self,
        image: str,
        scribbles: str,
        meta_key_postfix: str = "meta_dict",
        post_proc_label: str = "prob",
        scribbles_bg_label: int = 2,
        scribbles_fg_label: int = 3,
        model: str = "FEAT",
        loss: str = "CE",
        epochs: int = 80,
        lr: float = 0.1,
        lr_step: float = [0.1],
        lr_factor: float = 0.1, 
        dropout: float = 0.2,
        hidden_layers: List[int] = [16, 8],
        kernel_size: int = 9,
        num_filters: int = 128,
        train_feat: bool = True,
        use_argmax: bool = False,
        model_path: str = None,
        use_amp: bool = False,
        device: str = "cuda",
    ) -> None:
        super().__init__(meta_key_postfix)
        self.image = image
        self.scribbles = scribbles
        self.scribbles_bg_label = scribbles_bg_label
        self.scribbles_fg_label = scribbles_fg_label
        self.post_proc_label = post_proc_label
        self.model = model
        self.loss = loss
        self.epochs = epochs
        self.lr = lr
        self.lr_step = lr_step
        self.lr_factor = lr_factor
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.train_feat = train_feat
        self.use_argmax = use_argmax
        self.model_path = model_path
        self.use_amp = use_amp
        self.device = device

    def __call__(self, data):
        d = dict(data)

        # at the moment only supports binary seg problem
        num_classes = 2

        # copy affine meta data from image input
        d = self._copy_affine(d, src=self.image, dst=self.post_proc_label)

        # read relevant terms from data
        image = self._fetch_data(d, self.image)
        scribbles = self._fetch_data(d, self.scribbles)

        image = np.squeeze(image)
        scribbles = np.squeeze(scribbles)

        # zero-pad input image volume
        pad_size = int(math.floor(self.kernel_size / 2))
        image = np.pad(image, ((pad_size, pad_size),) * 3, mode="symmetric")

        # extract patches and select only relevant patches with scribble labels for online training
        image_patches = view_as_windows(
            image,
            (self.kernel_size, self.kernel_size, self.kernel_size),
            step=1,
        )

        # select relevant patches only for training network
        fg_patches = image_patches[scribbles == self.scribbles_fg_label]
        bg_patches = image_patches[scribbles == self.scribbles_bg_label]

        all_sel_patches = np.expand_dims(
            np.concatenate([fg_patches, bg_patches], axis=0), axis=1
        )
        all_sel_labels = np.concatenate(
            [
                np.ones((fg_patches.shape[0], 1, 1, 1, 1)),
                np.zeros((bg_patches.shape[0], 1, 1, 1, 1)),
            ],
        )

        # transfer data to pytorch tensors and relevant compute device
        image_patches_pt = (
            torch.from_numpy(all_sel_patches).type(torch.float32).to(device=self.device)
        )

        target_pt = (
            torch.from_numpy(all_sel_labels).type(torch.long).to(device=self.device)
        )
        logging.info(
            "Training using model features {} and loss {}".format(self.model, self.loss)
        )

        # load selected model module [ECONet (FEAT), ECONet-Haar (HAAR)]
        if self.model == "HAAR":
            # haar-like hand-crafted features ECONet-Haar-Like
            model = ECONetFCNHaarFeatures(
                kernel_size=self.kernel_size,
                hidden_layers=self.hidden_layers,
                num_classes=num_classes,
                haar_padding="valid",  # use valid padding as we need to reduce patches
                use_bn=True,
                activation=torch.nn.ReLU,
                dropout=self.dropout,
            ).to(device=self.device)
        elif self.model == "FEAT":
            # learned features ECONet (proposed)
            model = ECONetFCNLearnedFeatures(
                feat_kernel_size=self.kernel_size,
                feat_num_filters=self.num_filters,
                hidden_layers=self.hidden_layers,
                num_classes=num_classes,
                feat_padding="valid",
                use_bn=True,
                activation=torch.nn.ReLU,
                dropout=self.dropout,
            ).to(device=self.device)
        else:
            raise ValueError("Unknown model specified, {}".format(self.model))

        # if a model checkout found, load params
        if self.model_path and os.path.exists(self.model_path):
            # load
            logging.info("Loading online model from: %s" % self.model_path)
            try:
                model.load_state_dict(torch.load(self.model_path))
            except:
                # sometimes this may fail, particularly if the model has changed since last saved pt
                # this really is meant to be a temporary copy so okay to delete it in such cases
                logging.info(
                    "Unable to load weights, deleting previous model checkpoint at: {}".format(
                        self.model_path
                    )
                )
                os.unlink(self.model_path)

        if self.train_feat:
            params_to_train = [p for n, p in model.named_parameters()]
        else:
            # exclude haar parameters from learned parameters
            params_to_train = [
                p for n, p in model.named_parameters() if "featureextactor" not in n
            ]
            # haar is not to be learned, pre-compute features once and reuse to save compute
            with torch.no_grad():
                image_patches_pt = model(
                    image_patches_pt, skip_feat=False, skip_mlp=True
                )

        optim = torch.optim.Adam(params_to_train, lr=self.lr)
        # optim = torch.optim.RMSprop(params_to_train, lr=self.lr)
        # optim = torch.optim.SGD(params_to_train, lr=self.lr)

        # using single step at epochs * lr_step epoch to reduce lr by a factor of lr
        # e.g. epoch=50, lr_step=0.5 and lr=0.1, then lr=0.1 for epoch[0-25] and lr=0.01 for epoch[25-50]
        if self.lr_step:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optim,
                [int(self.epochs * lrstep) for lrstep in list(self.lr_step)],
                gamma=self.lr_factor,
            )

        # calculate weights for scribbles-balanced cross-entropy
        # help on imbalanced cross-entropy from:
        # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#calculate_class_weights
        number_of_samples = [np.sum(all_sel_labels == x) for x in range(num_classes)]

        # if even one class missing, then skip weighting
        skip_weighting = 0 in number_of_samples
        if not skip_weighting:
            eps = get_eps(image)
            weight_for_classes = (
                torch.tensor(
                    [
                        (1.0 / (x + eps))
                        * (sum(number_of_samples) / len(number_of_samples))
                        for x in number_of_samples
                    ]
                )
                .type(torch.float32)
                .to(self.device)
            )
        else:
            logging.info(
                "Skipping weighting for class, as atleast one class not in training data"
            )
            weight_for_classes = (
                torch.tensor([1.0] * len(number_of_samples))
                .to(torch.float32)
                .to(self.device)
            )
        logging.info("Samples per class:{}".format(number_of_samples))
        logging.info("Weights per class: {}".format(weight_for_classes))

        # load the loss function to used from [CrossEntropy (CE), DICE+CrossEntropy (DICECE)]
        reduction = "mean"
        if self.loss == "CE":
            loss_func = torch.nn.CrossEntropyLoss(
                weight=weight_for_classes, ignore_index=-1, reduction=reduction
            )
            target_pt = target_pt.squeeze(1)
        elif self.loss == "DICECE":
            loss_func = MyDiceCELoss(
                to_onehot_y=True,
                softmax=True,
                ce_weight=weight_for_classes,
                reduction=reduction,
            )
        else:
            raise ValueError("Invalid loss received {}".format(self.loss))

        # Automatic Mixed Precision (AMP) help from tutorial:
        # https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html#all-together-automatic-mixed-precision

        # amp is only used when use_amp=True, otherwise this is equivalent to fp32 pytorch training loop
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        model.train()
        pbar = tqdm.tqdm(range(self.epochs))
        for ep in pbar:
            # for idx, (patch_data, target_data) in enumerate(loader):
            # patch_data, target_data = patch_data.to(self.device), target_data.to(self.device)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                output_pt = model(image_patches_pt, skip_feat=not self.train_feat)
                loss = loss_func(output_pt, target_pt)
                del output_pt

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()
            pbar.set_description("Online Model Loss: %f" % loss.item())

            if self.lr_step:
                scheduler.step()

        # clear some GPU and CPU memory
        del (
            image_patches_pt,
            image_patches,
            target_pt,
            fg_patches,
            bg_patches,
            all_sel_labels,
            all_sel_patches,
        )

        # save model to storage if needed
        if self.model_path:
            logging.info("Saving online model to: %s" % self.model_path)
            torch.save(model.state_dict(), self.model_path)

        # perform inference using trained model
        # load full volume
        model.eval()
        image_pt = (
            torch.from_numpy(np.expand_dims(np.expand_dims(image, axis=0), axis=0))
            .type(torch.float32)
            .to(device=self.device)
        )

        # apply ECONet as Fully-Convolutional model to whole volume to infer full volume likelihood
        with torch.no_grad():
            try:
                output_pt = model(image_pt)
            except RuntimeError as e:
                if "out of memory" not in str(e):
                    raise RuntimeError(e)
                logging.info(str(e))
                logging.info("Not enough memory for online inference")
                logging.info("Trying inference using CPU (slower)")
                model = model.to("cpu")
                output_pt = model(image_pt.to("cpu"))

            output_pt = torch.softmax(output_pt, dim=1)

        # apply argmax to get predicted label (if needed)
        if self.use_argmax:
            output_pt = torch.argmax(output_pt, dim=1, keepdim=True).type(torch.float32)

        output_np = output_pt.squeeze(0).detach().cpu().numpy()

        d[self.post_proc_label] = output_np

        return d


class MakeLikelihoodFromScribblesDybaORFd(MyInteractiveSegmentationTransform):
    def __init__(
        self,
        image: str,
        scribbles: str,
        meta_key_postfix: str = "meta_dict",
        post_proc_label: str = "prob",
        scribbles_bg_label: int = 2,
        scribbles_fg_label: int = 3,
        kernel_size: int = 9,
        criterion: str = "entropy",
        num_trees: int = 50,
        max_tree_depth: int = 20,
        min_samples_split: int = 6,
        use_argmax: bool = False,
        model_path: str = None,
        device: str = "cuda",
    ) -> None:
        super().__init__(meta_key_postfix)
        self.image = image
        self.scribbles = scribbles
        self.scribbles_bg_label = scribbles_bg_label
        self.scribbles_fg_label = scribbles_fg_label
        self.post_proc_label = post_proc_label
        self.kernel_size = kernel_size
        self.criterion = criterion
        self.num_trees = num_trees
        self.max_tree_depth = max_tree_depth
        self.min_samples_split = min_samples_split
        self.use_argmax = use_argmax
        self.model_path = model_path
        self.device = device

    def __call__(self, data):
        d = dict(data)

        # at the moment only supports binary seg problem
        num_classes = 2

        # copy affine meta data from image input
        d = self._copy_affine(d, src=self.image, dst=self.post_proc_label)

        # read relevant terms from data
        image = self._fetch_data(d, self.image)
        scribbles = self._fetch_data(d, self.scribbles)

        # load haar-like feature extractor
        haar_feature_extractor = torchhaarfeatures.HaarFeatures3d(
            kernel_size=self.kernel_size,
            padding="same",
            stride=1,
            padding_mode="zeros",
        ).to(self.device)

        # extract features
        image_pt = torch.from_numpy(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = haar_feature_extractor(image_pt).cpu().numpy()

        features = np.squeeze(features)
        scribbles = np.squeeze(scribbles)

        inchannels, depth, height, width = features.shape

        # extract patches and select only relevant patches with scribble labels for online training
        features_patches = view_as_windows(features, (inchannels, 1, 1, 1), step=1)
        features_patches = np.squeeze(features_patches, axis=0)

        # select relevant patches only for training network
        fg_patches = features_patches[scribbles == self.scribbles_fg_label]
        bg_patches = features_patches[scribbles == self.scribbles_bg_label]

        all_sel_patches = np.concatenate([fg_patches, bg_patches], axis=0)
        all_sel_patches = np.squeeze(all_sel_patches)
        all_sel_labels = np.concatenate(
            [
                np.ones((fg_patches.shape[0])),
                np.zeros((bg_patches.shape[0])),
            ]
        )

        # calculate imbalance weights for DybaRF
        number_of_samples = [np.sum(all_sel_labels == x) for x in range(num_classes)]
        # if even one class missing, then skip weighting
        skip_weighting = 0 in number_of_samples
        if not skip_weighting:
            eps = get_eps(image)
            weight_for_classes = [
                (1.0 / (x + eps)) * (sum(number_of_samples) / len(number_of_samples))
                for x in number_of_samples
            ]
        else:
            logging.info(
                "Skipping weighting for class, as atleast one class not in training data"
            )
            weight_for_classes = [1.0] * len(number_of_samples)
        logging.info("Samples per class:{}".format(number_of_samples))
        logging.info("Weights per class: {}".format(weight_for_classes))

        logging.info(
            "Training RF using HAAR features num_trees={}, criterion={}".format(
                self.num_trees, self.criterion
            )
        )

        # load RandomForest module from sklearn
        model = RandomForestClassifier(
            n_estimators=self.num_trees,
            criterion=self.criterion,
            max_depth=self.max_tree_depth,
            min_samples_split=self.min_samples_split,
            class_weight={i: x for i, x in enumerate(weight_for_classes)},
        )

        # if a model found, load it
        if self.model_path and os.path.exists(self.model_path):
            logging.info("Loading online model from: %s" % self.model_path)
            try:
                with open(self.model_path, "rb") as fp:
                    model = pickle.load(fp)
            except:
                os.unlink(self.model_path)

        # train DybaORF model
        model.fit(all_sel_patches, all_sel_labels)

        # save model to storage if needed
        if self.model_path:
            logging.info("Saving online model to: %s" % self.model_path)
            with open(self.model_path, "wb") as fp:
                pickle.dump(model, fp)

        # perform inference using trained model
        # load features
        features = np.reshape(features, (16, -1)).transpose()

        # apply DybaORF to whole volume to infer full volume likelihood
        output = (
            model.predict_proba(features).transpose().reshape(-1, depth, height, width)
        )
        output = output.astype(np.float32)

        # apply argmax to get predicted label (if needed)
        if self.use_argmax:
            output = np.expand_dims(np.argmax(output, axis=0), axis=0).astype(
                np.float32
            )

        d[self.post_proc_label] = output

        return d


class MakeLikelihoodFromScribblesGMMd(MyInteractiveSegmentationTransform):
    def __init__(
        self,
        image: str,
        scribbles: str,
        meta_key_postfix: str = "meta_dict",
        post_proc_label: str = "prob",
        scribbles_bg_label: int = 2,
        scribbles_fg_label: int = 3,
        mixture_size: int = 20,
        normalise: bool = True,
    ) -> None:
        super().__init__(meta_key_postfix)
        self.image = image
        self.scribbles = scribbles
        self.scribbles_bg_label = scribbles_bg_label
        self.scribbles_fg_label = scribbles_fg_label
        self.post_proc_label = post_proc_label
        self.mixture_size = mixture_size
        self.normalise = normalise

    def __call__(self, data):
        d = dict(data)

        # copy affine meta data from image input
        d = self._copy_affine(d, src=self.image, dst=self.post_proc_label)

        # read relevant terms from data
        image = self._fetch_data(d, self.image)
        scribbles = self._fetch_data(d, self.scribbles)

        # make likelihood image
        post_proc_label = make_likelihood_image_gmm(
            image,
            scribbles,
            scribbles_bg_label=self.scribbles_bg_label,
            scribbles_fg_label=self.scribbles_fg_label,
            return_label=False,
            mixture_size=self.mixture_size,
        )

        if self.normalise:
            post_proc_label = self._normalise_logits(post_proc_label, axis=0)

        d[self.post_proc_label] = post_proc_label

        return d


class MakeLikelihoodFromScribblesHistogramd(MyInteractiveSegmentationTransform):
    def __init__(
        self,
        image: str,
        scribbles: str,
        meta_key_postfix: str = "meta_dict",
        post_proc_label: str = "prob",
        scribbles_bg_label: int = 2,
        scribbles_fg_label: int = 3,
        normalise: bool = True,
        alpha_bg: int = 1,
        alpha_fg: int = 1,
        bins: int = 32,
    ) -> None:
        super().__init__(meta_key_postfix)
        self.image = image
        self.scribbles = scribbles
        self.scribbles_bg_label = scribbles_bg_label
        self.scribbles_fg_label = scribbles_fg_label
        self.post_proc_label = post_proc_label
        self.normalise = normalise
        self.alpha_bg = alpha_bg
        self.alpha_fg = alpha_fg
        self.bins = bins

    def __call__(self, data):
        d = dict(data)

        # copy affine meta data from image input
        d = self._copy_affine(d, src=self.image, dst=self.post_proc_label)

        # read relevant terms from data
        image = self._fetch_data(d, self.image)
        scribbles = self._fetch_data(d, self.scribbles)

        # make likelihood image
        post_proc_label = make_likelihood_image_histogram(
            image,
            scribbles,
            scribbles_bg_label=self.scribbles_bg_label,
            scribbles_fg_label=self.scribbles_fg_label,
            return_label=False,
            alpha_bg=self.alpha_bg,
            alpha_fg=self.alpha_fg,
            bins=self.bins,
        )

        if self.normalise:
            post_proc_label = self._normalise_logits(post_proc_label, axis=0)

        d[self.post_proc_label] = post_proc_label

        return d


class ApplyGaussianSmoothing(MyInteractiveSegmentationTransform):
    def __init__(
        self,
        image: str,
        meta_key_postfix="meta_dict",
        kernel_size: int = 3,
        sigma: float = 1.0,
        device: str = "cuda",
    ):
        super().__init__(meta_key_postfix)
        self.image = image
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.device = device

    def __call__(self, data):
        d = dict(data)

        # read relevant terms from data
        image = self._fetch_data(d, self.image)

        # determine dimensionality of input tensor
        spatial_dims = len(image.shape) - 1

        # add batch dimension
        image = torch.from_numpy(image).to(self.device).unsqueeze_(0)

        # initialise smoother
        if spatial_dims == 2:
            smoother = GaussianSmoothing2d(
                in_channels=image.shape[1],
                kernel_size=self.kernel_size,
                sigma=self.sigma,
                padding="same",
                stride=1,
                padding_mode="zeros",
            ).to(self.device)
        elif spatial_dims == 3:
            smoother = GaussianSmoothing3d(
                in_channels=image.shape[1],
                kernel_size=self.kernel_size,
                sigma=self.sigma,
                padding="same",
                stride=1,
                padding_mode="zeros",
            ).to(self.device)
        else:
            raise ValueError(
                "Gaussian smoothing not defined for {}-dimensional input".format(
                    spatial_dims
                )
            )

        # apply smoothing
        image = smoother(image).squeeze(0).detach().cpu().numpy()

        d[self.image] = image

        return d


class Timeit(MyInteractiveSegmentationTransform):
    def __init__(
        self,
        time_key: str = "time",
        tic_key: str = "tic",
        meta_key_postfix: str = "meta_dict",
    ) -> None:
        super().__init__(meta_key_postfix)
        self.time_key = time_key
        self.tic_key = tic_key

    def __call__(self, data):
        d = dict(data)

        tic = d.get(self.tic_key, None)
        if tic is not None:
            new_tic = time.time()
            toc = new_tic - tic
            time_list = d.get(self.time_key, [])
            time_list.append(toc)
            d[self.time_key] = time_list
            d[self.tic_key] = new_tic
            print()
            print("-" * 80)
            print("Time Taken: {}".format(time_list))
            print("-" * 80)
            print()
        else:
            d[self.tic_key] = time.time()

        return d

class AddBackgroundScribblesFromROIWithDropfracd(MyInteractiveSegmentationTransform):
    def __init__(
        self,
        scribbles: str,
        roi_key: str = "roi",
        meta_key_postfix: str = "meta_dict",
        scribbles_bg_label: int = 2,
        scribbles_fg_label: int = 3,
        drop_frac: float = 0.9,
    ) -> None:
        super().__init__(meta_key_postfix)
        self.scribbles = scribbles
        self.roi_key = roi_key
        self.scribbles_bg_label = scribbles_bg_label
        self.scribbles_fg_label = scribbles_fg_label
        self.drop_frac = drop_frac

    def __call__(self, data):
        d = dict(data)

        # read relevant terms from data
        scribbles = self._fetch_data(d, self.scribbles)

        # get any existing roi information and apply it to scribbles, skip otherwise
        selected_roi = d.get(self.roi_key, None)
        if selected_roi:
            mask = np.ones_like(scribbles).astype(np.bool)
            mask[
                :,
                selected_roi[0] : selected_roi[1],
                selected_roi[2] : selected_roi[3],
                selected_roi[4] : selected_roi[5],
            ] = 0

            if self.drop_frac:
                drop_mask = np.random.uniform(size=mask.shape)
                drop_mask = (drop_mask >= self.drop_frac)

                mask = mask & drop_mask


            # prune outside roi region as bg scribbles
            scribbles[mask] = self.scribbles_bg_label

            # if no foreground scribbles found, then add a scribble at center of roi
            if not np.any(scribbles == self.scribbles_fg_label):
                # issue a warning - the algorithm should still work
                logging.info(
                    "warning: no foreground scribbles received with label {}, adding foreground scribbles to ROI centre".format(
                        self.scribbles_fg_label
                    )
                )
                offset = 5

                cx = int((selected_roi[0] + selected_roi[1]) / 2)
                cy = int((selected_roi[2] + selected_roi[3]) / 2)
                cz = int((selected_roi[4] + selected_roi[5]) / 2)

                # add scribbles at center of roi
                scribbles[
                    :, cx - offset : cx + offset, cy - offset : cy + offset, cz - offset : cz + offset
                ] = self.scribbles_fg_label

        # return new scribbles
        d[self.scribbles] = scribbles

        return d

class ApplyGraphCutOptimisationd(InteractiveSegmentationTransform):
    """
    Generic GraphCut optimisation transform.

    This can be used in conjuction with any Make*Unaryd transform
    (e.g. MakeISegUnaryd from above for implementing ISeg unary term).
    It optimises a typical energy function for interactive segmentation methods using numpymaxflow's GraphCut method,
    e.g. Equation 5 from https://arxiv.org/pdf/1710.04043.pdf.

    Usage Example::

        Compose(
            [
                # unary term maker
                MakeISegUnaryd(
                    image="image",
                    logits="logits",
                    scribbles="label",
                    unary="unary",
                    scribbles_bg_label=2,
                    scribbles_fg_label=3,
                ),
                # optimiser
                ApplyGraphCutOptimisationd(
                    unary="unary",
                    pairwise="image",
                    post_proc_label="pred",
                    lamda=10.0,
                    sigma=15.0,
                ),
            ]
        )
    """

    def __init__(
        self,
        unary: str,
        pairwise: str,
        meta_key_postfix: str = "meta_dict",
        post_proc_label: str = "pred",
        lamda: float = 8.0,
        sigma: float = 0.1,
    ) -> None:
        super().__init__(meta_key_postfix)
        self.unary = unary
        self.pairwise = pairwise
        self.post_proc_label = post_proc_label
        self.lamda = lamda
        self.sigma = sigma

    def __call__(self, data):
        d = dict(data)

        # attempt to fetch algorithmic parameters from app if present
        self.lamda = d.get("lamda", self.lamda)
        self.sigma = d.get("sigma", self.sigma)

        # copy affine meta data from pairwise input
        self._copy_affine(d, self.pairwise, self.post_proc_label)

        # read relevant terms from data
        unary_term = self._fetch_data(d, self.unary)
        pairwise_term = self._fetch_data(d, self.pairwise)

        # check if input unary is compatible with GraphCut opt
        if unary_term.shape[0] > 2:
            raise ValueError(
                "GraphCut can only be applied to binary probabilities, received {}".format(unary_term.shape[0])
            )

        # # attempt to unfold probability term
        # unary_term = self._unfold_prob(unary_term, axis=0)

        # prepare data for numpymaxflow's GraphCut
        # run GraphCut
        post_proc_label = maxflow(pairwise_term, unary_term, lamda=self.lamda, sigma=self.sigma)

        d[self.post_proc_label] = post_proc_label

        return d