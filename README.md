### Interactive framework for evaluation of CNN architectures


#### How to run   

---
##### Poetry environment
Start indiviudual environment session  
`poetry shell`

Close indiviudual environment session    
`exit`

---
##### Train/evaluate

Train resnet18 model   
`python runner.py --mode train --model resnet18 --epochs 5`

Evaluate resnet18 model  
`python runner.py --mode eval --model resnet18`   

`python runner.py --mode eval --model resnet18 --weights 
checkpoints/resnet18/epoch_5.pth`


---
##### Visualize:   
random 2:   
`python visualize.py --models resnet18`  

indices:   
`poetry run python visualize.py --models resnet18 --indices 12 45 102`

compare models:   
`python visualize.py --models baseline,resnet18 --indices 12 45`

compare models random:   
`python visualize.py --models baseline,resnet18 --num_samples 3`

---
##### Plot
plot results:   
`python plot_results.py --models baseline`

plot results:   
`python plot_results.py --models baseline,resnet18`

---




#### Project structure
```
exp_1/
├── config.py     
├── dataset.py         # data loader
├── models/
│   ├── factory.py     # model factory
│   ├── baseline.py
│   └── resnet_siam.py
├── utils/
│   ├── metrics.py    
│   └── trainer.py   
└── runner.py          # entry point
```