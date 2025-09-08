import torch
import lightning as L
from ema_pytorch.ema_pytorch import EMA

class BaseWrapper(L.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        loss_params: dict[str, Any],
        opt_params: dict[str, Any],
        sched_params: dict[str, Any] | None = None,
        ema_params: dict[str, Any] | None = None,
    ):
        super().__init__()

        self.backbone = backbone
        self.loss_params = loss_params
        self.opt_params = opt_params
        self.sched_params = sched_params
        self.ema_params = ema_params

        self.save_hyperparameters()


    def configure_model(self):
        self.model = build_model(self.model_params, show=False)
        self.loss = instantiate_from_mapping(self.loss_params)

    def probe_mode(self, head : nn.Module):
        for param in self.parameters():
            param.requires_grad = False
        self.probe_head = head
        self.probe_head.requires_grad = True


    def init_optimizer_from_config(self, config: dict[str, Any]):
        ## if config is a list, then we have multiple optimizers
        ## if target_params is in a config then the optimizer will only optimize those parameters
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
        self.projection_head = projection_head
    
    def training_step(self, batch, batch_idx):
        target_sims = batch.get("target_sims")
        views = batch.get("views")

        out_ = self.backbone(views)
        z = out_["z"]
        g = self.projection_head(z)

        loss = self.loss(z, target_sims)
        self.log("loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        target_sims = batch.get("target_sims")
        views = batch.get("views")

        out_ = self.backbone(views)
        z = out_["z"]
        g = self.projection_head(z)

        loss = self.loss(g, target_sims)
        self.log("loss", loss)
        return loss


class SimSiam(BaseWrapper):

    def __init__(self, backbone: nn.Module, projection_head: nn.Module, predictor_head: nn.Module, loss_params: dict[str, Any], opt_params: dict[str, Any], sched_params: dict[str, Any] | None = None, ema_params: dict[str, Any] | None = None):
        super().__init__(backbone, loss_params, opt_params, sched_params, ema_params)
        self.projection_head = projection_head
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
        self.projection_head = projection_head
        self.predictor_head = predictor_head

        
    def configure_ema(self):
        ## make an ema model of all modules
        self.ema_backbone = copy.deepcopy(self.backbone)
        self.ema_projection_head = copy.deepcopy(self.projection_head)
        self.ema_predictor_head = copy.deepcopy(self.predictor_head)


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
            h_stop = self.ema_predictor_head(g_stop)

        h_1, h_2 = torch.chunk(h, 2, dim = 0)

        g_stop_1, g_stop_2 = torch.chunk(g_stop, 2, dim = 0)

        loss = self.loss(h_1, g_stop_2) + self.loss(h_2, g_stop_1)

        self.log("loss", loss)
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
            h_stop = self.ema_predictor_head(g_stop)

        h_1, h_2 = torch.chunk(h, 2, dim = 0)
        g_stop_1, g_stop_2 = torch.chunk(g_stop, 2, dim = 0)

        loss = self.loss(h_1, g_stop_2) + self.loss(h_2, g_stop_1)

        self.log("loss", loss)
        return loss


class BarlowTwins(BaseWrapper):
    def __init__(self, backbone: nn.Module, projection_head: nn.Module, loss_params: dict[str, Any], opt_params: dict[str, Any], sched_params: dict[str, Any] | None = None, ema_params: dict[str, Any] | None = None):
        super().__init__(backbone, loss_params, opt_params, sched_params, ema_params)
        self.projection_head = projection_head
        
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

