import logging

logger = logging.getLogger(__name__)

from monai.transforms import (Compose, EnsureChannelFirstd, LoadImaged,
                              ScaleIntensityRanged, Spacingd)
from monailabel.interfaces.tasks.infer import InferTask, InferType
from monailabel.scribbles.transforms import AddBackgroundScribblesFromROId
from monailabel.transform.post import BoundingBoxd, Restored

from lib.transforms import (AddBackgroundScribblesFromROIWithDropfracd,
                            ApplyGaussianSmoothing, ApplyGraphCutOptimisationd,
                            MakeLikelihoodFromScribblesDybaORFd,
                            MakeLikelihoodFromScribblesECONetd,
                            MakeLikelihoodFromScribblesGMMd,
                            MakeLikelihoodFromScribblesHistogramd, Timeit)


class MyLikelihoodBasedSegmentor(InferTask):
    def __init__(
        self,
        dimension=3,
        description="Generic base class for constructing online likelihood based segmentors",
        intensity_range=(-1000, 400, 0.0, 1.0, True),
        pix_dim=(2.0, 2.0, 2.0),
        lamda=5.0,
        sigma=0.1,
        config=None,
    ):
        super().__init__(
            path=None,
            network=None,
            labels="region 7",
            type=InferType.SCRIBBLES,
            dimension=dimension,
            description=description,
            config=config,
        )
        self.intensity_range = intensity_range
        self.pix_dim = pix_dim
        self.lamda = lamda
        self.sigma = sigma

    def pre_transforms(self):
        return [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            # AddBackgroundScribblesFromROId(
            AddBackgroundScribblesFromROIWithDropfracd(
                scribbles="label", scribbles_bg_label=2, scribbles_fg_label=3, drop_frac=0.98
            ),
            Spacingd(
                keys=["image", "label"],
                pixdim=self.pix_dim,
                mode=["bilinear", "nearest"],
            ),
            ScaleIntensityRanged(
                keys="image",
                a_min=self.intensity_range[0],
                a_max=self.intensity_range[1],
                b_min=self.intensity_range[2],
                b_max=self.intensity_range[3],
                clip=self.intensity_range[4],
            ),
            ApplyGaussianSmoothing(
                image="image",
                kernel_size=3,
                sigma=1.0,
                device="cuda",
            ),
        ]

    def post_transforms(self):
        return [
            ApplyGraphCutOptimisationd(
                unary="prob",
                pairwise="image",
                post_proc_label="pred",
                lamda=self.lamda,
                sigma=self.sigma,
            ),
            Timeit(),
            Restored(keys="pred", ref_image="image"),
            BoundingBoxd(keys="pred", result="result", bbox="bbox"),
        ]


class ECONetPlusGraphCut(MyLikelihoodBasedSegmentor):
    """
    Defines Efficient Convolutional Online Likelihood Network (ECONet) based Online Likelihood training and inference method for
    COVID-19 lung lesion segmentation based on the following paper:

    Asad, Muhammad, Lucas Fidon, and Tom Vercauteren. "" ECONet: Efficient Convolutional Online Likelihood Network
    for Scribble-based Interactive Segmentation."
    To be reviewed (preprint: https://arxiv.org/pdf/2201.04584.pdf).

    This task takes as input 1) original image volume and 2) scribbles from user
    indicating foreground and background regions. A likelihood volume is learned and inferred using ECONet method.

    numpymaxflow's GraphCut layer is used to regularise the resulting likelihood, where unaries come from likelihood
    and pairwise is the original input volume.

    This also implements variations of ECONet with hand-crafted features, referred as ECONet-Haar-Like in the paper.
    """

    def __init__(
        self,
        dimension=3,
        description="Online likelihood inference with ECONet for COVID-19 lung lesion segmentation",
        intensity_range=(-1000, 400, 0.0, 1.0, True),
        pix_dim=(2.0, 2.0, 2.0),
        lamda=5.0,
        sigma=0.1,
        model="FEAT",
        loss="CE",
        epochs=200,
        lr=0.01,
        lr_step=[0.7],
        dropout=0.3,
        hidden_layers=[32, 16],
        kernel_size=7,
        num_filters=128,
        train_feat=True,
        model_path=None,
        config=None,
    ):
        super().__init__(
            dimension=dimension,
            description=description,
            intensity_range=intensity_range,
            pix_dim=pix_dim,
            lamda=lamda,
            sigma=sigma,
            config=config,
        )
        self.model = model
        self.loss = loss
        self.epochs = epochs
        self.lr = lr
        self.lr_step = lr_step
        self.dropout = dropout
        self.hidden_layers = hidden_layers
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.train_feat = train_feat
        self.model_path = model_path

    def inferer(self):
        return Compose(
            [
                Timeit(),
                MakeLikelihoodFromScribblesECONetd(
                    image="image",
                    scribbles="label",
                    post_proc_label="prob",
                    scribbles_bg_label=2,
                    scribbles_fg_label=3,
                    model=self.model,
                    loss=self.loss,
                    epochs=self.epochs,
                    lr=self.lr,
                    lr_step=self.lr_step,
                    dropout=self.dropout,
                    hidden_layers=self.hidden_layers,
                    kernel_size=self.kernel_size,
                    num_filters=self.num_filters,
                    train_feat=self.train_feat,
                    use_argmax=False,
                    model_path=self.model_path,
                    use_amp=False,
                    device="cuda",
                ),
                Timeit(),
            ]
        )


class DybaORFPlusGraphCut(MyLikelihoodBasedSegmentor):
    """
    Defines Dynamically Balanced Online Random Forest (DybaORF) based Online Likelihood training and inference method for
    COVID-19 lung lesion segmentation based on the following paper:

    Wang, Guotai, et al. "Dynamically balanced online random forests for interactive scribble-based segmentation."
    International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2016.

    This task takes as input 1) original image volume and 2) scribbles from user
    indicating foreground and background regions. A likelihood volume is learned and inferred using DybaORF-Haar-Like method.

    numpymaxflow's GraphCut layer is used to regularise the resulting likelihood, where unaries come from likelihood
    and pairwise is the original input volume.
    """

    def __init__(
        self,
        dimension=3,
        description="Online likelihood inference with DybaORF-Haar for COVID-19 lung lesion segmentation",
        intensity_range=(-1000, 400, 0.0, 1.0, True),
        pix_dim=(2.0, 2.0, 2.0),
        lamda=5.0,
        sigma=0.1,
        kernel_size=9,
        criterion="entropy",
        num_trees=50,
        max_tree_depth=20,
        min_samples_split=6,
        model_path=None,
        config=None,
    ):
        super().__init__(
            dimension=dimension,
            description=description,
            intensity_range=intensity_range,
            pix_dim=pix_dim,
            lamda=lamda,
            sigma=sigma,
            config=config,
        )
        self.kernel_size = kernel_size
        self.criterion = criterion
        self.num_trees = num_trees
        self.max_tree_depth = max_tree_depth
        self.min_samples_split = min_samples_split
        self.model_path = model_path

    def inferer(self):
        return Compose(
            [
                Timeit(),
                MakeLikelihoodFromScribblesDybaORFd(
                    image="image",
                    scribbles="label",
                    post_proc_label="prob",
                    scribbles_bg_label=2,
                    scribbles_fg_label=3,
                    kernel_size=self.kernel_size,
                    criterion=self.criterion,
                    num_trees=self.num_trees,
                    max_tree_depth=self.max_tree_depth,
                    min_samples_split=self.min_samples_split,
                    use_argmax=False,
                    model_path=self.model_path,
                    device="cuda",
                ),
                Timeit(),
            ]
        )


class GMMPlusGraphCut(MyLikelihoodBasedSegmentor):
    """
    Defines Gaussian Mixture Model (GMM) based Online Likelihood generation method for COVID-19 lung lesion segmentation based on the following paper:

    Rother, Carsten, Vladimir Kolmogorov, and Andrew Blake. "" GrabCut" interactive foreground extraction using iterated graph cuts."
    ACM transactions on graphics (TOG) 23.3 (2004): 309-314.

    This task takes as input 1) original image volume and 2) scribbles from user
    indicating foreground and background regions. A likelihood volume is generated using GMM method.

    numpymaxflow's GraphCut layer is used to regularise the resulting likelihood, where unaries come from likelihood
    and pairwise is the original input volume.
    """

    def __init__(
        self,
        dimension=3,
        description="Online likelihood generation using GMM for COVID-19 lung lesion segmentation",
        intensity_range=(-1000, 400, 0.0, 1.0, True),
        pix_dim=(2.0, 2.0, 2.0),
        lamda=5.0,
        sigma=0.1,
        mixture_size=20,
        config=None,
    ):
        super().__init__(
            dimension=dimension,
            description=description,
            intensity_range=intensity_range,
            pix_dim=pix_dim,
            lamda=lamda,
            sigma=sigma,
            config=config,
        )
        self.mixture_size = mixture_size

    def inferer(self):
        return Compose(
            [
                Timeit(),
                MakeLikelihoodFromScribblesGMMd(
                    image="image",
                    scribbles="label",
                    post_proc_label="prob",
                    scribbles_bg_label=2,
                    scribbles_fg_label=3,
                    mixture_size=self.mixture_size,
                ),
                Timeit(),
            ]
        )


class HistogramPlusGraphCut(MyLikelihoodBasedSegmentor):
    """
    Defines Histogram-based Online Likelihood generation method for COVID-19 lung lesion segmentation based on the following paper:

    Boykov, Yuri Y., and M-P. Jolly. "Interactive graph cuts for optimal boundary & region segmentation of objects in ND images."
    Proceedings eighth IEEE international conference on computer vision. ICCV 2001. Vol. 1. IEEE, 2001.

    This task takes as input 1) original image volume and 2) scribbles from user
    indicating foreground and background regions. A likelihood volume is generated using histogram method.

    numpymaxflow's GraphCut layer is used to regularise the resulting likelihood, where unaries come from likelihood
    and pairwise is the original input volume.
    """

    def __init__(
        self,
        dimension=3,
        description="Online likelihood generation using Histogram for COVID-19 lung lesion segmentation",
        intensity_range=(-1000, 400, 0.0, 1.0, True),
        pix_dim=(2.0, 2.0, 2.0),
        lamda=5.0,
        sigma=0.1,
        alpha_bg=1,
        alpha_fg=1,
        bins=128,
        config=None,
    ):
        super().__init__(
            dimension=dimension,
            description=description,
            intensity_range=intensity_range,
            pix_dim=pix_dim,
            lamda=lamda,
            sigma=sigma,
            config=config,
        )
        self.alpha_bg = alpha_bg
        self.alpha_fg = alpha_fg
        self.bins = bins

    def inferer(self):
        return Compose(
            [
                Timeit(),
                MakeLikelihoodFromScribblesHistogramd(
                    image="image",
                    scribbles="label",
                    post_proc_label="prob",
                    scribbles_bg_label=2,
                    scribbles_fg_label=3,
                    normalise=True,
                    alpha_bg=self.alpha_bg,
                    alpha_fg=self.alpha_fg,
                    bins=self.bins,
                ),
                Timeit(),
            ]
        )
