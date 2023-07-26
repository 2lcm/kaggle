import wandb

api = wandb.Api({"entity":"2lcm"})
run = api.run('asl/model_03_2')

files = run.files()
for file in files:
    print(file)