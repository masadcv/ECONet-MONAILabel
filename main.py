import logging
import os
from typing import Dict

from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.tasks.infer import InferTask

from lib.scribbles import (
    GMMPlusGraphCut,
    HistogramPlusGraphCut,
    DybaORFPlusGraphCut,
    ECONetPlusGraphCut,
)

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies, conf):

        # path to save model for current sample
        model_dir = os.path.join(app_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        self.online_model_path = os.path.join(model_dir, "ECONetOnline.pt")

        # clear existing ECONet weights that may be present from previous experiments
        if os.path.exists(self.online_model_path):
            os.unlink(self.online_model_path)

        # intensity and spacing options
        self.intensity_range = (-1000, 400, 0.0, 1.0, True)
        self.pix_dim = (2.0, 2.0, 2.0)

        # GraphCut optimisation parameters
        self.lamda = 5.0
        self.sigma = 0.1

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            name="Segmentation - ECONet based AI-Assisted Annotation",
            description="ECONet: Efficient Convolutional Online likelihood Network for AI-Assisted annotation of 3D CT Images from COVID-19 patients",
        )

    def init_infers(self) -> Dict[str, InferTask]:
        return {
            "ECONet+GraphCut": ECONetPlusGraphCut(
                intensity_range=self.intensity_range,
                pix_dim=self.pix_dim,
                model="FEAT",
                loss="CE",
                train_feat=True,
                lamda=self.lamda,
                sigma=self.sigma,
                model_path=self.online_model_path,
            ),
            "ECONet-Haar+GraphCut": ECONetPlusGraphCut(
                intensity_range=self.intensity_range,
                pix_dim=self.pix_dim,
                model="HAAR",
                loss="CE",
                train_feat=False,
                lamda=self.lamda,
                sigma=self.sigma,
                model_path=self.online_model_path,
            ),
            "DybaORF-Haar+GraphCut": DybaORFPlusGraphCut(
                intensity_range=self.intensity_range,
                pix_dim=self.pix_dim,
                lamda=self.lamda,
                sigma=self.sigma,
                model_path=self.online_model_path,
            ),
            "GMM+GraphCut": GMMPlusGraphCut(
                intensity_range=self.intensity_range,
                pix_dim=self.pix_dim,
                lamda=self.lamda,
                sigma=self.sigma,
            ),
            "Histogram+GraphCut": HistogramPlusGraphCut(
                intensity_range=self.intensity_range,
                pix_dim=self.pix_dim,
                lamda=self.lamda,
                sigma=self.sigma,
            ),
        }

    def next_sample(self, request):
        # clear ECONet weights when new sample is loaded
        if hasattr(self, "online_model_path") and os.path.exists(
            self.online_model_path
        ):
            logging.info(
                "Clearing online model for previous sample: {}".format(
                    self.online_model_path
                )
            )
            os.unlink(self.online_model_path)

        return super().next_sample(request)
