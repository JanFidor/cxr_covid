import wandb
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def initialize_wandb(experiment_name, group, project, config):
    return wandb.init(
        name=experiment_name,
        group=group,
        project=project,
        config=config
    )

def log_metrics(stage, epoch, epoch_loss):
    log_dict = {f"loss/{stage}": epoch_loss,}
    wandb.log(log_dict, epoch=epoch)

def log_confmat(stage, epoch, confmat, class_names):
    f, ax = plt.subplots(figsize = (15,10)) 
    plot = sn.heatmap(
        confmat,
        annot=True, ax=ax,
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plot.set_xticklabels(class_names, rotation=45)
    plot.set_yticklabels(class_names)
    
    wandb.log({
        f"confmat/{stage}" : wandb.Image(f)
    }, epoch=epoch)
    plt.close()
        
def print_classwise_metric(stage, metric_name, metric, class_names):
    print(f"{stage} {metric_name} values:")
    for i, c in enumerate(class_names):
        print(f"{c}: {round(metric[i].item(), 4)}")
    print()

def log_image(_type, stage, image, class_name, epoch, logged_idx):
    image_wandb = wandb.Image(image, caption=f"{class_name}")
    wandb.log({f"{_type}/{stage}/{class_name}/{logged_idx}": image_wandb}, step=epoch)


