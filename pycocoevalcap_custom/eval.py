__author__ = 'tylin'
from .tokenizer.ptbtokenizer import PTBTokenizer,NLTKTokenizer
from .bleu.bleu import Bleu
from .meteor.meteor import Meteor,MeteorHF
from .rouge.rouge import Rouge
from .cider.cider import Cider
from .spice.spice import Spice
import logging
import nltk


class COCOEvalCap:
    def __init__(self, coco, cocoRes):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.coco = coco
        self.cocoRes = cocoRes
        self.params = {'image_id': coco.getImgIds()}
        self.logger = logging.getLogger(__name__)
        #self.logger.info("Download NLTK packages")
        self.logger.info("Append NLTK packages path")
        #nltk.download('punkt', quiet=True)
        #nltk.download('wordnet', quiet=True)
        #nltk.download('omw-1.4', quiet=True)
        nltk.data.path.append('/home/guests/sebastian_sotela/nltk_data')
    
    def evaluate(self):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.cocoRes.imgToAnns[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        self.logger.info('Setting up NLTK Tokenizer...')
        # FIXME using nltk for running on IFL slurm which
        # has no java
        # tokenizer = PTBTokenizer()
        tokenizer = NLTKTokenizer()
        self.logger.info("Tokenizing for evaluation...")
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        self.logger.info("Setting up scorers...")
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            # FIXME: Using hugging face implementation since java
            # is not available in ifl cluster
            # (Meteor(),"METEOR"),
            (MeteorHF(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            #(Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
      
        for scorer, method in scorers:
            self.logger.info(f"Computing Score for method {method}...")
            try:
                score, scores = scorer.compute_score(gts, res)
            except Exception as e:
                self.logger.error(f"Problem scoring with method {method}: {e}")

            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    self.logger.info(f"{m}: {sc:.3f}")
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                self.logger.info(f"{method}: {score:.3f}")
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]
