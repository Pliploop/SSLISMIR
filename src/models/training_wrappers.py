import torch
import lightning as L
from ema_pytorch.ema_pytorch import EMA
from typing import Any
from torch import nn
import copy

try:
    from utils.utils import instantiate_from_mapping
    from utils.utils import build_model
except ImportError: # notebook
    from src.utils.utils import instantiate_from_mapping
    from src.utils.utils import build_model
    
    

class BaseWrapper(L.LightningModule):
    def __init__(
        self,
        model_params,
        loss_params,
        opt_params,
        sched_params,
        ema_params,
    ):
        super().__init__()

        if isinstance(model_params, nn.Module):
            self.backbone = model_params
        else:
            self.backbone = build_model(model_params)

        if isinstance(loss_params, dict):
            self.loss = instantiate_from_mapping(loss_params)
        else:
            self.loss = loss_params
        
        # if the optimizers, schedulers are not lists or dicts, they are just a single optimizer or scheduler
        
        if isinstance(sched_params, list) or isinstance(sched_params, dict):
            self.sched_params = sched_params
        else:
            self.sched_params = [sched_params]
        
        self.sched_params = sched_params
        self.ema_params = ema_params
        self.opt_params = opt_params



    def probe_mode(self, head : nn.Module):
        for param in self.parameters():
            param.requires_grad = False
        self.probe_head = head
        self.probe_head.requires_grad = True


    def init_optimizer_from_config(self, config: dict[str, Any]):
    
        ## if config is a list, then we have multiple optimizers
        ## if target_params is in a config then the optimizer will only optimize those parameters
        
        if not isinstance(config, list) and not isinstance(config, dict):
            return config
        
        if isinstance(config, list):
            opts = []
            for opt_config in config:
                if "target_params" in opt_config:
                    params = self.get_parameters_from_pattern(opt_config.pop("target_params"))
                    print(f"Optimizing {len(params)} parameters for {opt_config}")
                    opts.append(instantiate_from_mapping(opt_config, params=params))
                else:
                    opts.append(instantiate_from_mapping(opt_config, params=self.model.parameters()))
            return opts
        else:
            if "target_params" in config:
                params = self.get_parameters_from_pattern(config.pop("target_params"))
                print(f"Optimizing {len(params)} parameters for {config}")
                return [instantiate_from_mapping(config, params=params)]
            else:
                return [instantiate_from_mapping(config, params=self.model.parameters())]
            

    def configure_optimizers(self):
        opt = self.init_optimizer_from_config(self.opt_params)
        
        
        if self.sched_params is not None:
            if isinstance(self.sched_params, list):
                scheds = []
                ## check that the opt length matches the sched length
                if len(opt) != len(self.sched_params):
                    raise ValueError(f"The number of optimizers ({len(opt)}) does not match the number of schedulers ({len(self.sched_params)})")
                for sched_config, opt_ in zip(self.sched_params, opt):
                    scheds.append({
                        "scheduler": instantiate_from_mapping(sched_config, optimizer=opt_),
                        "interval": "step",
                    })
                return opt, scheds
            else:
                sched = [instantiate_from_mapping(self.sched_params, optimizer=opt_) for opt_ in opt]
                return opt, sched
        return opt

    def get_parameters_from_pattern(self, pattern: str):
        return [p for n, p in self.named_parameters() if n.startswith(pattern)]


class ContrastiveLearning(BaseWrapper):

    def __init__(self, backbone: nn.Module, projection_head: nn.Module, loss_params: dict[str, Any], opt_params: dict[str, Any], sched_params: dict[str, Any] | None = None, ema_params: dict[str, Any] | None = None):
        super().__init__(backbone, loss_params, opt_params, sched_params, ema_params)
        if isinstance(projection_head, nn.Module):
            self.projection_head = projection_head
        else:
            self.projection_head = build_model(projection_head)
            
        self.model = nn.ModuleDict({
            "backbone": self.backbone,
            "projection_head": self.projection_head
        })
    
    def training_step(self, batch, batch_idx):
        target_sims = batch.get("target_sims")
        views = batch.get("views")

        out_ = self.backbone(views)
        z = out_["z"]
        g = self.projection_head(z)
        

        loss = self.loss(g, target_sims)
        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        target_sims = batch.get("target_sims")
        views = batch.get("views")

        out_ = self.backbone(views)
        z = out_["z"]
        g = self.projection_head(z)

        loss = self.loss(g, target_sims)
        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        return loss


class SimSiam(BaseWrapper):

    def __init__(self, backbone: nn.Module, projection_head: nn.Module, predictor_head: nn.Module, loss_params: dict[str, Any], opt_params: dict[str, Any], sched_params: dict[str, Any] | None = None, ema_params: dict[str, Any] | None = None):
        super().__init__(backbone, loss_params, opt_params, sched_params, ema_params)
        if isinstance(projection_head, nn.Module):
            self.projection_head = projection_head
        else:
            self.projection_head = build_model(projection_head)
        if isinstance(predictor_head, nn.Module):
            self.predictor_head = predictor_head
        else:
            self.predictor_head = build_model(predictor_head)
        self.predictor_head = predictor_head

    def training_step(self, batch, batch_idx):
        views = batch.get("views")

        out_ = self.backbone(views)
        z = out_["z"]
        g = self.projection_head(z)
        h = self.predictor_head(g)

        with torch.no_grad():
            out_stop = self.backbone(views)
            z_stop = out_stop["z"]
            g_stop = self.projection_head(z_stop)

        h_1, h_2 = torch.chunk(h, 2, dim = 0)
        g_stop_1, g_stop_2 = torch.chunk(g_stop, 2, dim = 0)

        loss = self.loss(h_1, g_stop_2) + self.loss(h_2, g_stop_1)

        self.log("loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        target_sims = batch.get("target_sims")
        views = batch.get("views")

        out_ = self.model(views)
        z = out_["z"]
        g = self.projection_head(z)
        h = out_["h"]

        with torch.no_grad():
            out_stop = self.backbone(views)
            z_stop = out_stop["z"]
            g_stop = out_stop["g"]



        h_1, h_2 = torch.chunk(h, 2, dim = 0)
        g_stop_1, g_stop_2 = torch.chunk(g_stop, 2, dim = 0)

        loss = self.loss(h_1, g_stop_2) + self.loss(h_2, g_stop_1)

        self.log("loss", loss)
        return loss
    

class BYOL(BaseWrapper):
    ## same thing as SimSiam but the stop gradient predictions are with the ema model
    ## and the stop gradient targets are with the model

    def __init__(self, backbone: nn.Module, projection_head: nn.Module, predictor_head: nn.Module, loss_params: dict[str, Any], opt_params: dict[str, Any], sched_params: dict[str, Any] | None = None, ema_params: dict[str, Any] | None = None):
        super().__init__(backbone, loss_params, opt_params, sched_params, ema_params)
        if isinstance(projection_head, nn.Module):
            self.projection_head = projection_head
        else:
            self.projection_head = build_model(projection_head)
        if isinstance(predictor_head, nn.Module):
            self.predictor_head = predictor_head
        else:
            self.predictor_head = build_model(predictor_head)

        self.model = nn.ModuleDict({
            "backbone": self.backbone,
            "projection_head": self.projection_head,
            "predictor_head": self.predictor_head
        })

    def on_fit_start(self):
        self.configure_ema()

    def configure_ema(self):
        ## make an ema model of all modules
        self.ema_backbone = copy.deepcopy(self.backbone)
        self.ema_projection_head = copy.deepcopy(self.projection_head)
        self.ema_model = nn.ModuleDict({
            "backbone": self.ema_backbone,
            "projection_head": self.ema_projection_head,
        })

    def training_step(self, batch, batch_idx):
        target_sims = batch.get("target_sims")
        views = batch.get("views")

        out_ = self.backbone(views)
        z = out_["z"]
        g = self.projection_head(z)
        h = self.predictor_head(g)

        with torch.no_grad():
            out_stop = self.ema_backbone(views)
            z_stop = out_stop["z"]
            g_stop = self.ema_projection_head(z_stop)

        h_1, h_2 = torch.chunk(h, 2, dim = 0)

        g_stop_1, g_stop_2 = torch.chunk(g_stop, 2, dim = 0)

        loss = self.loss(h_1, g_stop_2) + self.loss(h_2, g_stop_1)

        self._maybe_update_ema(self.global_step)

        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        target_sims = batch.get("target_sims")
        views = batch.get("views")

        out_ = self.backbone(views)
        z = out_["z"]
        g = self.projection_head(z)
        h = self.predictor_head(g)

        with torch.no_grad():
            out_stop = self.ema_backbone(views)
            z_stop = out_stop["z"]
            g_stop = self.ema_projection_head(z_stop)

        h_1, h_2 = torch.chunk(h, 2, dim = 0)
        g_stop_1, g_stop_2 = torch.chunk(g_stop, 2, dim = 0)

        loss = self.loss(h_1, g_stop_2) + self.loss(h_2, g_stop_1)

        self.log("loss", loss)
        return loss

    def _maybe_update_ema(self, global_step: int):
        # if step > update_after_step and step % update_every == 0
        if global_step > self.ema_params.get("update_after_step", 0) and global_step % self.ema_params.get("update_every", 1) == 0:
            self.update_ema()

    def update_ema(self):
        rate = self.ema_params.get("beta", 0.996)
        for param, ema_param in zip(self.backbone.parameters(), self.ema_backbone.parameters()):
            ema_param.data.copy_(rate * ema_param.data + (1 - rate) * param.data)
        for param, ema_param in zip(self.projection_head.parameters(), self.ema_projection_head.parameters()):
            ema_param.data.copy_(rate * ema_param.data + (1 - rate) * param.data)


class BarlowTwins(BaseWrapper):
    def __init__(self, backbone: nn.Module, projection_head: nn.Module, loss_params: dict[str, Any], opt_params: dict[str, Any], sched_params: dict[str, Any] | None = None, ema_params: dict[str, Any] | None = None):
        super().__init__(backbone, loss_params, opt_params, sched_params, ema_params)
        if isinstance(projection_head, nn.Module):
            self.projection_head = projection_head
        else:
            self.projection_head = build_model(projection_head)
        
    def training_step(self, batch, batch_idx):
        views = batch.get("views")

        out_ = self.backbone(views)
        z = out_["z"]
        g = self.projection_head(z)
        
        
        ## chunk the batch into two views
        z_1, z_2 = torch.chunk(z, 2, dim = 0)

        loss = self.loss(z_1, z_2)

        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        views = batch.get("views")

        out_ = self.backbone(views)
        z = out_["z"]
        g = self.projection_head(z)
        
        ## chunk the batch into two views
        z_1, z_2 = torch.chunk(z, 2, dim = 0)

        loss = self.loss(z_1, z_2)

        self.log("loss", loss)
        return loss


class Supervised(BaseWrapper):
    def __init__(self, backbone: nn.Module, projection_head: nn.Module, loss_params: dict[str, Any], opt_params: dict[str, Any], sched_params: dict[str, Any] | None = None, ema_params: dict[str, Any] | None = None):
        super().__init__(backbone, loss_params, opt_params, sched_params, ema_params)
        if isinstance(projection_head, nn.Module):
            self.projection_head = projection_head
        else:
            self.projection_head = build_model(projection_head)
            
    def training_step(self, batch, batch_idx):
        audio = batch.get("audio")

        out_ = self.backbone(views)
        z = out_["z"]
        g = self.projection_head(z)

        loss = self.loss(g, batch.get("labels"))
        return loss