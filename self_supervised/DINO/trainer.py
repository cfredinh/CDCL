import pdb
import wandb
from self_supervised.BYOL.trainer import *
        
class DINOTrainer(BYOLTrainer):
    def __init__(self, wraped_defs, freeze_last_for=1,final_weight_decay=0.4,stop_early=0):
        super().__init__(wraped_defs, stop_early=stop_early) 
        self.freeze_last_for = freeze_last_for
        self.stop_early = stop_early
        self.decay_scheduler = CosineSchedulerWithWarmup(
            base_value=self.optimizer.param_groups[0]["weight_decay"], 
            final_value=final_weight_decay, iters=len(self.trainloader)*self.epochs)  

        
    def global_step(self, **kwargs):
        self.optimizer.zero_grad()
        self.aux_optimizer.zero_grad()
        aux_loss = None
        # get batch
        images, labels = kwargs['batch']  
        if len(labels) == 2 and isinstance(labels, list):
            ids    = labels[1]
            labels = labels[0]
            
        # go through the model
        with autocast(self.use_mixed_precision):
            loss, aux_outs = self.model(images, epoch = self.epoch-1, domain_belonging = ids) 
        # backprop
        if aux_outs is not None:
            aux_loss = self.criterion(aux_outs, labels.to(self.device_id, non_blocking=True))
            if not self.use_mixed_precision:
                aux_loss.backward() 
                self.aux_optimizer.step()           
            else:
                self.scaler.scale(aux_loss).backward()
                self.scaler.step(self.aux_optimizer)
                self.scaler.update()     
        
        if not self.use_mixed_precision:
            loss.backward() 
            if self.grad_clipping:
                clipped_params = (w for n,w in self.model.named_parameters() if "aux_fc" not in n)
                torch.nn.utils.clip_grad_norm_(clipped_params, self.grad_clipping)
            if self.epoch <= self.freeze_last_for:
                cancel_gradients(self.model, "student_encoder.fc.last_layer")                
            self.optimizer.step()  
        else:
            self.scaler.scale(loss).backward()
            if self.grad_clipping:
                self.scaler.unscale_(self.optimizer)
                clipped_params = (w for n,w in self.model.named_parameters() if "aux_fc" not in n)
                torch.nn.utils.clip_grad_norm_(clipped_params, self.grad_clipping)
            if self.epoch <= self.freeze_last_for:
                cancel_gradients(self.model, "student_encoder.fc.last_layer")                
            self.scaler.step(self.optimizer)
            self.scaler.update() 


        if ddp_is_on():
            self.model.module.ema_update(self.iters)
        else:
            self.model.ema_update(self.iters)

        # updating lr and wd
        self.scheduler.step(self.val_target, self.val_loss)
        if aux_outs is not None and self.aux_scheduler is not None:
            self.aux_scheduler.step(self.val_target, self.val_loss)
        self.optimizer.param_groups[0]["weight_decay"] = self.decay_scheduler(self.iters)
        if self.iters % self.log_every == 0 or (self.iters == 1 and not self.is_grid_search):
            loss = dist_average_tensor(loss)
            if self.is_rank0:
                log_dict = {'train_loss': loss.item(),
                            'learning_rate': self.get_lr()}
                if aux_outs is not None:
                    log_dict['aux_learning_rate'] = self.aux_optimizer.param_groups[0]['lr']
                self.logging(log_dict) 
                if aux_loss is not None:
                    self.logging({'aux_loss': aux_loss.item()})                     
                        
    @property
    def feature_extractor(self):
        return DINO_to_classifier(self.model)
                
def DINO_to_classifier(net):
    if is_parallel(net):
        return net.module.teacher_encoder
    else:
        return net.teacher_encoder      
