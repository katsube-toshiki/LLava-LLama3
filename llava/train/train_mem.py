from llava.train.train import train
import wandb

if __name__ == "__main__":
    wandb.login(key="5b626f60267bea23db2ae2bb33b5b3289f936cb1")
    train(attn_implementation="flash_attention_2")
