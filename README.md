### Interactive framework for evaluation of CNN architectures


#### How to run   


Start indiviudual environment session  
`poetry shell`

Train resnet18 model   
`python runner.py --mode train --model resnet18 --epochs 5`

Evaluate resnet18 model  
`python runner.py --mode eval --model resnet18`
`python runner.py --mode eval --model resnet18 --weights checkpoints/resnet18/epoch_5.pth`

Close indiviudual environment session    
`exit`



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