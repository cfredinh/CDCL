import wandb
from defaults.trainer import *
        
class BYOLTrainer(Trainer):
    def __init__(self, wraped_defs, use_momentum=True, stop_early=0):
        super().__init__(wraped_defs)
        self.use_momentum = use_momentum
        self.stop_early = stop_early
        self.best_model = model_to_CPU_state(self.feature_extractor)        
        self.validate_learning_on_single_batch = False
        
    def train(self):
        self.test_mode = False
        self.print_train_init()
        
        epoch_bar = range(self.epoch0 + 1, self.epoch0 + self.epochs + 1)
        if self.is_rank0:
            epoch_bar = tqdm(epoch_bar, desc='Epoch', leave=False)
        
        # Looping thorough epochs
        for self.epoch in epoch_bar:
            if isinstance(self.trainloader.sampler, DS):
                self.trainloader.sampler.set_epoch(self.epoch)            
            self.model.train()             
            iter_bar = enumerate(self.trainloader)
            if self.is_rank0:
                iter_bar = tqdm(iter_bar, desc='Training', leave=False, total=len(self.trainloader))
            
            # Looping through batches
            if self.validate_learning_on_single_batch:
                for it, batch in iter_bar:
                    break
                    
                for it in range(int(len(iter_bar))):
                    self.iters += 1
                    print(self.iters)
                    self.global_step(batch=batch, it=it)   

                    # going through epoch step
                    if self.val_every != np.inf:
                        if (self.iters % int(self.val_every * self.epoch_steps) == 0): 
                            synchronize()
                            self.epoch_step()  
                            self.model.train()    
                    synchronize()   
                
            else:
                for it, batch in iter_bar:
                    self.iters += 1
                    self.global_step(batch=batch, it=it)   

                    # going through epoch step
                    if self.val_every != np.inf:
                        if (self.iters % int(self.val_every * self.epoch_steps) == 0): 
                            synchronize()
                            self.epoch_step()  
                            self.model.train()    
                    synchronize()   
                
            if not self.save_best_model and not self.is_grid_search:
                self.best_model = model_to_CPU_state(self.feature_extractor)
                self.save_session()      
                
            if self.use_aux_head and self.reset_aux_every:
                if self.epoch < self.epochs + 1 and self.epoch % self.reset_aux_every == 0:
                    if is_ddp(self.model):
                        if self.model.module.student_encoder.aux_fc is not None:
                            self.model.module.student_encoder.aux_fc.reset()                        
                        if self.model.module.teacher_encoder.aux_fc is not None:                        
                            self.model.module.teacher_encoder.aux_fc.reset()
                    else:
                        if self.model.student_encoder.aux_fc is not None:
                            self.model.student_encoder.aux_fc.reset()                        
                        if self.model.teacher_encoder.aux_fc is not None:                        
                            self.model.teacher_encoder.aux_fc.reset()
            print(self.epoch, self.stop_early) 
            if self.stop_early > 0 and self.stop_early == self.epoch:
                break

        print_ddp(" ==> Training done")
        if not self.is_grid_search:
            self.save_session(verbose=True)
        synchronize()
        
    def global_step(self, **kwargs):
        self.optimizer.zero_grad()
        # get batch
        images, labels = kwargs['batch']  
        if len(labels) == 2 and isinstance(labels, list):
            ids    = labels[1]
            labels = labels[0]

        # go through the model
        with autocast(self.use_mixed_precision):
            loss, aux_outs = self.model(images) 

        if aux_outs is not None:
            aux_labels = labels.to(self.device_id, non_blocking=True)
            aux_loss = self.criterion(aux_outs, aux_labels)
            if not self.use_mixed_precision:
                aux_loss.backward() 
                self.aux_optimizer.step()           
            else:
                self.scaler.scale(aux_loss).backward()
                self.scaler.step(self.aux_optimizer)
                self.scaler.update()          

        # backprop
        if not self.use_mixed_precision:
            loss.backward() 
            if self.grad_clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clipping)
            self.optimizer.step()  
        else:
            self.scaler.scale(loss).backward()
            if self.grad_clipping:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clipping)
            self.scaler.step(self.optimizer)
            self.scaler.update()  
        
        if aux_outs is not None and self.aux_scheduler is not None:
            self.aux_scheduler.step(self.val_target, self.val_loss)

        if self.use_momentum:
            if ddp_is_on():
                self.model.module.ema_update(self.iters)
            else:
                self.model.ema_update(self.iters)
            

        self.scheduler.step(self.val_target, self.val_loss)
        if self.iters % self.log_every == 0 or (self.iters == 1 and not self.is_grid_search):
            loss = dist_average_tensor(loss)
            if self.is_rank0:
                self.logging({'train_loss': loss.item(),
                             'learning_rate': self.get_lr()}) 
                if aux_outs is not None:
                    self.logging({'aux_learning_rate' : self.aux_optimizer.param_groups[0]['lr']}) 
                 

    def epoch_step(self, **kwargs):  
        self.val_iters += 1
        knn_eval = self.knn_eval_every and (self.val_iters % self.knn_eval_every) == 0
        self.evaluate(knn_eval=knn_eval) 
        
        if not self.is_grid_search:
            self.save_session()        
     
    def evaluate(self, dataloader=None, knn_eval=False, prefix='val', 
                 calc_train_feats=True, val_taget='knn', **kwargs):
        """Validation loop function.            
        This is pretty much the same thing with global_step() but with torch.no_grad()
        Also note that DDP is not used here. There is not much point to DDP, since 
        we are not doing backprop anyway.
        """
        used_val_for_best = f"{val_taget}_{prefix}"
        if knn_eval and calc_train_feats:
            self.build_feature_bank()
            
        if not self.is_rank0: return
        # Note: I am removing DDP from evaluation since it is slightly slower 
        self.model.eval()
        if dataloader == None:
            dataloader = self.valloader
            
        if not len(dataloader):
            self.best_model = model_to_CPU_state(self.feature_extractor)
            self.model.train()
            return

        n_classes = dataloader.dataset.n_classes
        knn_nhood = dataloader.dataset.knn_nhood
        target_metric = dataloader.dataset.target_metric
        aux_metric, knn_metric = None, None
        if self.is_rank0:
            if knn_eval:
                knn_metric = self.metric_fn(n_classes, dataloader.dataset.int_to_labels, mode=f"knn_{prefix}")
            if self.use_aux_head:
                aux_metric = self.metric_fn(n_classes, dataloader.dataset.int_to_labels, mode=f"aux_{prefix}")
            iter_bar = tqdm(dataloader, desc='Validating', leave=False, total=len(dataloader))
        else:
            iter_bar = dataloader
        
        self.val_loss = None
        aux_val_loss = []
        feature_bank = []
        id_bank      = []
        with torch.no_grad():
            ijk = 0
            for images, labels in iter_bar:
                if len(labels) == 2 and isinstance(labels, list):
                    ids    = labels[1]
                    labels = labels[0]
                elif len(labels) == 3 and isinstance(labels, list):
                    ids    = labels[2]
                    labels = labels[0]
                    

                labels = labels.to(self.device_id, non_blocking=True)
                images = images.to(self.device_id, non_blocking=True)

                if is_ddp(self.model):
                    _, features, aux_outs = self.model.module(images, return_embedding=True)
                else:
                    _, features, aux_outs = self.model(images, return_embedding=True)                  

                if self.log_embeddings:
                    feature_bank.append(features.clone().detach().cpu())
                    id_bank.append(ids)

                # knn_eval
                if knn_eval:
                    features = F.normalize(features, dim=1)
                    pred_labels = self.knn_predict(feature = features, 
                                                   feature_bank = self.feature_bank, 
                                                   feature_labels =  self.targets_bank, 
                                                   knn_k = knn_nhood, knn_t = 0.1, classes=n_classes,
                                                   multi_label = not dataloader.dataset.is_multiclass)
                    knn_metric.add_preds(pred_labels, labels, using_knn=True)
                if aux_metric is not None:
                    aux_loss = self.criterion(aux_outs, labels)
                    aux_val_loss.append(aux_loss.item())                    
                    aux_metric.add_preds(aux_outs, labels)         

                ijk += 1

                #if ijk == 50:
                #    break        

        # building Umap embeddings        
        if self.log_embeddings and knn_eval:
            self.build_umaps(feature_bank, dataloader, labels=knn_metric.truths,  id_bank=id_bank, mode=f'{prefix}')

        eval_metrics = edict({})
        if knn_eval:
            knn_metric = knn_metric.get_value(use_dist=isinstance(dataloader,DS))
            eval_metrics.update(knn_metric)
            setattr(self, f"knn_{prefix}_target", knn_metric[f"knn_{prefix}_{target_metric}"])
        if aux_metric is not None:
            aux_val_loss = np.array(aux_val_loss).mean()
            aux_metric = aux_metric.get_value(use_dist=isinstance(dataloader,DS))
            eval_metrics.update(aux_metric)
            setattr(self, f"aux_{prefix}_target", aux_metric[f"aux_{prefix}_{target_metric}"])
        if used_val_for_best == self.used_type_for_save_best:
            self.val_target = getattr(self, f"{used_val_for_best}_target")
        
        if not self.is_grid_search:
            if self.report_intermediate_steps:
                self.logging(eval_metrics)
                if aux_metric is not None:
                    self.logging({f'aux_{prefix}_loss': round(aux_val_loss, 5)})                
            if self.val_target > self.best_val_target:
                self.best_val_target = self.val_target
                if self.save_best_model:
                    self.best_model = model_to_CPU_state(self.feature_extractor)
            if not self.save_best_model:
                self.best_model = model_to_CPU_state(self.feature_extractor)
        self.model.train()

                    
    @property
    def feature_extractor(self):
        return BYOL_to_classifier(self.model)
                
def BYOL_to_classifier(net):
    if is_parallel(net):
        return net.module.online_encoder
    else:
        return net.online_encoder        
