import torch.nn as nn
import pytorch_lightning as pl
import tabpfn
import dl_approach as dla
import torch

class tabularModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.tabPFN = dla.load_model('/vol/chameleon/projects/adni/adni_1/trained_models/tabular_baseline.pth')
        self.tabPFN_model = self.tabPFN.model[2]

        #self.tabPFN_backbone = nn.Sequential(*list(self.tabPFN.model[2].children())[:-1][:-2])
        #self.head = nn.Sequential(nn.GELU(approximate='none'), nn.Linear(1024, 3, bias = True))

        for param in self.tabPFN_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        output = self.tabPFN_model(x)
        return output


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.head.parameters, lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        pred = self.forward(x)
        loss = torch.nn.CrossEntropyLoss(pred,y)
        logs = {"train_loss":loss}
        return {"loss": loss, "log": logs}



    trainer = pl.Trainer()







