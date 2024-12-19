
import gc
import numpy as np 
import torch
import torch.nn as nn 
import torch.optim as optim
from os import path
from tqdm import tqdm
from models.metrics import Challenge2018Score
from sklearn.metrics import cohen_kappa_score, accuracy_score
#%% 
class TrainerForPrototype:

    def __init__(self, model, device, lr=1e-5,
        tensorboard=True, context_manager=True ):

        # Model
        self.device = device 
        self.model = model.to(self.device)

        # Loss Function 
        self.criterion_arous = nn.CrossEntropyLoss(ignore_index=-1, reduction='none').to(self.device)
        self.criterion_apnea = nn.CrossEntropyLoss(ignore_index=-1, reduction='none').to(self.device)
        self.criterion_sleep = nn.CrossEntropyLoss(ignore_index=-1, reduction='none').to(self.device)

        # Optimizer 
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=lr)

        # Training Utils 
        self.tensorboard = tensorboard
        self.context_manager = context_manager

        self.epochs = 0
        if self.context_manager: 
            self.history = self.__load_context()
        else:
            self.history = self.__init_history()

    def fit_generator(self, gen_train, gen_valid=False, epochs=1):

        # Loop over epochs  
        for epoch_i in range(self.epochs, epochs+self.epochs):   

            # Training Loop over batch
            pbar = tqdm(gen_train)
            metric_trains = []
            for batch_x, batch_y, batch_w in pbar:

                # Allocate Batch data to device
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_w = batch_w.to(self.device)

                # Feed Forward and Back propagation
                metric_trains.append(self.step_on_batch(batch_x, batch_y, batch_w))

                # Progress bar에 Update
                pbar.set_description(self.__description( 
                    epoch=f"{epoch_i} TRAIN ",
                    epoch_metric=self.__avg_metrics(metric_trains)))

            # 최종 Training metric 계산
            metric_trains = self.__avg_metrics(metric_trains)

            # Validation Loop over batch
            if gen_valid:
                pbar = tqdm(gen_valid)
                metric_valids = []
                for batch_x, batch_y, batch_w in pbar:

                    # Allocate Batch data to device
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    batch_w = batch_w.to(self.device)

                    # Feed Forward and Back propagation
                    metric_valids.append(self.step_on_batch(batch_x, batch_y, batch_w, is_valid=True))

                    # Progress bar에 Update
                    pbar.set_description(self.__description( 
                        epoch=f"{epoch_i} VALID >>",
                        epoch_metric=self.__avg_metrics(metric_valids)))

                # 최종 Validation metric 계산
                metric_valids= self.__avg_metrics(metric_valids) 
            
            #Update History
            self.__update_history(
                train_metric=metric_trains, 
                valid_metric=metric_valids)

            #Update Tensorboard
            if self.tensorboard:
                self.tensorboard.write_epoch(
                    idx=epoch_i, history=self.history)

            #Update context_manager
            if self.context_manager:
                self.context_manager.save({
                    "epoch"               : epoch_i,
                    "history"             : self.history,
                    'model_state_dict'    : self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, name = "training_context.tar")
                
                self.context_manager.save(
                    self.model, name = f"model_{epoch_i}.pt")

    def step_on_batch(self, inputs, targets, weights=None, is_valid=False):

        if is_valid:
            no_grad = torch.no_grad()
            no_grad.__enter__()
        else: 
            # init gradation value of optimizer as zero
            self.optimizer.zero_grad()
        # labels init 
        target_arous = targets[:,0,:].contiguous().view(-1).long().to(self.device)
        target_apnea = targets[:,1,:].contiguous().view(-1).long().to(self.device)
        target_sleep = targets[:,2,:].contiguous().view(-1).long().to(self.device)

        # weights init
        if isinstance(weights, torch.Tensor):
            weight_arous = weights[:,0,:].contiguous().view(-1).to(self.device)
            weight_apnea = weights[:,1,:].contiguous().view(-1).to(self.device)
            weight_sleep = weights[:,2,:].contiguous().view(-1).to(self.device)

        # Feed Forward
        output_arous, output_apnea, output_sleep = self.model(inputs)
        output_arous = output_arous.permute(0, 2, 1).contiguous().view(-1, 2)
        output_apnea = output_apnea.permute(0, 2, 1).contiguous().view(-1, 2)
        output_sleep = output_sleep.permute(0, 2, 1).contiguous().view(-1, 2)
        # if True: # Debug 
        #     print(f"target_arous : {target_arous.shape}")
        #     print(f"output_arous : {output_arous.shape}")
        # Calculate Auxiliary loss
        loss_arous = self.criterion_arous(output_arous, target_arous)
        loss_apnea = self.criterion_apnea(output_apnea, target_apnea)
        loss_sleep = self.criterion_sleep(output_sleep, target_sleep)

        # Apply weight to loss 
        if isinstance(weights, torch.Tensor):
            loss_arous = torch.mean(loss_arous * weight_arous)
            loss_apnea = torch.mean(loss_apnea * weight_apnea)
            loss_sleep = torch.mean(loss_sleep * weight_sleep)
        else:
            loss_arous = torch.mean(loss_arous)
            loss_apnea = torch.mean(loss_apnea)
            loss_sleep = torch.mean(loss_sleep)

        # Calcualte total loss 
        loss = ((2*loss_arous) + loss_apnea + loss_sleep) / 4.0

        # Backpropagation 
        if is_valid: 
            no_grad.__exit__()
        else: # if it is training loop
            loss.backward()
            self.optimizer.step()


        # Calculate metrics

        metrics = self.__calc_metric(
            target_arous, target_apnea, target_sleep,
            output_arous, output_apnea, output_sleep, is_valid=is_valid)

        apendix = 'val_' if is_valid else ''
        metrics[f"{apendix}loss"] = loss.detach().cpu().numpy()
        
        return metrics

    def __description(self, epoch, epoch_metric):

        description = f"EPOCH: {epoch}>"
        
        for name, val in epoch_metric.items():
            if 'val_' in name:
                description += f"{name.replace('val_', '')}: {val:.4f} "
            else:
                description += f"{name}: {val:.4f} "

        return description

    def __calc_metric(self, target_arous, target_apnea, target_sleep,
        output_arous, output_apnea, output_sleep, is_valid=False):
        
        # to numpy 
        output_arous = torch.exp(output_arous).detach().cpu().numpy() 
        output_apnea = torch.exp(output_apnea).detach().cpu().numpy()
        output_sleep = torch.exp(output_sleep).detach().cpu().numpy()

        target_arous = target_arous.detach().cpu().numpy()
        target_apnea = target_apnea.detach().cpu().numpy()
        target_sleep = target_sleep.detach().cpu().numpy()

        # Calculate the probabilities for each target
        prob_arous = np.squeeze(output_arous[::, 1])
        prob_apnea = np.squeeze(output_apnea[::, 1])
        prob_sleep = np.squeeze(output_sleep[::, 0])

        # Calculate the predictions for each target
        pred_arous = np.apply_along_axis(func1d=np.argmax, axis=1, arr=output_arous)
        pred_apnea = np.apply_along_axis(func1d=np.argmax, axis=1, arr=output_arous)
        pred_sleep = np.apply_along_axis(func1d=np.argmax, axis=1, arr=output_arous)

        # Remove ignore index from arousal
        prob_arous   = prob_arous[target_arous >= 0]
        pred_arous   = pred_arous[target_arous >= 0]
        target_arous = target_arous[target_arous >= 0]
        prob_arous[prob_arous > 0.999] = 0.999
        prob_arous[prob_arous < 0.001] = 0.001

        # Remove ignore index from apnea
        prob_apnea = prob_apnea[target_apnea >= 0]
        pred_apnea = pred_apnea[target_apnea >= 0]
        target_apnea = target_apnea[target_apnea >= 0]
        prob_apnea[prob_apnea > 0.999] = 0.999
        prob_apnea[prob_apnea < 0.001] = 0.001

        # Remove ignore index from sleep
        prob_sleep = prob_sleep[target_sleep >= 0]
        pred_sleep = pred_sleep[target_sleep >= 0]
        target_sleep = target_sleep[target_sleep >= 0]
        prob_sleep[prob_sleep > 0.999] = 0.999
        prob_sleep[prob_sleep < 0.001] = 0.001

        # Compute performance stats
        scorer = Challenge2018Score()
        scorer.score_record(truth=target_arous, predictions=prob_arous)
        ar_auc, ar_aup = scorer._auc(scorer._pos_values, scorer._neg_values)
        ar_kpa = cohen_kappa_score(target_arous, pred_arous)
        # ar_acc = accuracy_score(target_arous, pred_arous)

        scorer = Challenge2018Score()
        scorer.score_record(truth=target_apnea, predictions=prob_apnea)
        ap_auc, ap_aup = scorer._auc(scorer._pos_values, scorer._neg_values)
        ap_kpa = cohen_kappa_score(target_apnea, pred_apnea)
        # ap_acc = accuracy_score(target_apnea, pred_apnea)

        scorer = Challenge2018Score()
        scorer.score_record(truth=1-target_sleep, predictions=prob_sleep)
        sl_auc, sl_aup = scorer._auc(scorer._pos_values, scorer._neg_values)
        sl_kpa = cohen_kappa_score(target_sleep, pred_sleep)
        # sl_acc = accuracy_score(target_sleep, pred_sleep)
        
        apendix = 'val_' if is_valid else ''
        metrics = {
            f"{apendix}ar_aup":ar_aup, f"{apendix}ar_kpa":ar_kpa, 
            f"{apendix}ap_aup":ap_aup, f"{apendix}ap_kpa":ap_kpa, 
            f"{apendix}sl_aup":sl_aup, f"{apendix}sl_kpa":sl_kpa, }
        # metrics = {
        #     f"{apendix}ar_auc":ar_auc, f"{apendix}ar_aup":ar_aup, f"{apendix}ar_kpa":ar_kpa, f"{apendix}ar_acc":ar_acc,
        #     f"{apendix}ap_auc":ap_auc, f"{apendix}ap_aup":ap_aup, f"{apendix}ap_kpa":ap_kpa, f"{apendix}ap_acc":ap_acc,
        #     f"{apendix}sl_auc":sl_auc, f"{apendix}sl_aup":sl_aup, f"{apendix}sl_kpa":sl_kpa, f"{apendix}sl_acc":sl_acc }

        gc.collect()

        return metrics

    def __avg_metrics(self, list_metric):

        # init avg_metrics functions
        keys = list(list_metric[0].keys())
        cal_avg = lambda list_metric, key: np.nanmean([ 
            metric[key] for metric in list_metric])

        return { key : cal_avg(list_metric, key) for key in keys }

    def __init_history(self):

        
        history = dict()
        his_keys = ["loss", 
             "ar_aup", "ar_kpa",  
             "ap_aup", "ap_kpa",  
             "sl_aup", "sl_kpa"]
        
        his_keys += [ "val_"+key for key in his_keys]

        history = dict([(key,[]) for key in his_keys])
 

        return history

    def __update_history(self, train_metric, valid_metric=None):

        for key, val in train_metric.items():
            self.history[key].append(val)
        
        if valid_metric != None:
            for key, val in valid_metric.items():
                self.history[key].append(val)
        else:
            for key in self.history:
                if 'val_' in key:
                    self.history[key].append(0)

    def __load_context(self,):
        
        context_file = path.join(self.context_manager.root,"training_context.tar")
        if path.isfile(context_file):

            context = self.context_manager.load("training_context.tar")

            self.epochs  = context["epoch"]
            self.history = context["history"]
            self.model.load_state_dict(context['model_state_dict'])
            self.optimizer.load_state_dict(context['optimizer_state_dict'])
            
            self.model.to(self.device)
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
        
            return self.history

        else:
            return self.__init_history()







#%% 