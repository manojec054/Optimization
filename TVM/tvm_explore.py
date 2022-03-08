from ast import arg
from numpy import dtype
import torch
import torchvision
import torchvision.transforms as transforms
import os
import time
import argparse
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
sys.path.append("~/tvm/python")

import os
os.environ["PATH"] = os.environ["PATH"]+":/usr/local/cuda/bin/"

import tvm
from tvm import relay
from tvm.contrib import graph_executor

assert(torch.cuda.is_available())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class MetaData():
    def __init__(self, test_batch, model_name, tvm_enable) -> None:
        pass

        # Number of distinct number labels, [0..9]
        self.NUM_CLASSES = 10
        self.samples = 100000
        self.seed = 10
        self.width = 32
        self.height = 32
        self.tvm_enable = tvm_enable

        # Number of examples in each training batch (step)
        self.TRAIN_BATCH_SIZE = 16
        self.TEST_BATCH = test_batch
        self.epochs = 10
        self.test_iteration = 1

        # Number of training steps to run
        self.TRAIN_STEPS = 1000

        self.model_path = './saved_model'

        # Loads CIFAR10 dataset.
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  

        self.sample_to_take = 10000

        if self.TEST_BATCH > 50:
            self.sample_to_take = 2000


        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=self.transform)
        
        self.trainsubset = torch.utils.data.Subset(self.trainset, range(0, 100))
        self.trainloader = torch.utils.data.DataLoader(self.trainsubset, batch_size=self.TRAIN_BATCH_SIZE,
                                                shuffle=True, num_workers=2)

        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=self.transform)
        self.testsubset = torch.utils.data.Subset(self.testset, range(0,self.sample_to_take))
        self.testloader = torch.utils.data.DataLoader(self.testsubset, batch_size=self.TEST_BATCH,
                                                shuffle=False, num_workers=2)

        self.classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        self.model_name = model_name
        self.model_path = f'./saved_model/cifar_net_{self.model_name}.pt'
        if self.tvm_enable:
            self.tvm_status = 'enabled'
        else:
            self.tvm_status = 'disabled'

class ModelPool():
    def __init__(self, params) -> None:
        self.params = params

    def create_vgg16(self):
        model = torchvision.models.vgg16(pretrained=True)
        return model.to(device)

    def create_resnet50(self):
        model = torchvision.models.resnet50(pretrained=True)
        return model.to(device)

    def create_resnet101(self):        
        model = torchvision.models.resnet101(pretrained=True)
        return model.to(device)


    def generate_model(self):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 6, 5)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, 5)
                self.fc1 = nn.Linear(16 * 5 * 5, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)

            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = torch.flatten(x, 1)  # flatten all dimensions except batch
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x
        
        model = Net().to(device)
        return model

class PlayGround():
    def __init__(self, params:MetaData) -> None:
        self.params = params

    def train(self, model):
        global model_name
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        for epoch in tqdm(range(self.params.epochs)):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(self.params.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
                    
                break
                    
        input_shape = [self.params.TRAIN_BATCH_SIZE, 3, self.params.width, self.params.height]
        input_data = torch.randn(input_shape).to(device)
        scripted_model = torch.jit.trace(model, input_data).eval()
        #scripted_model = torch.jit.script(model)
        scripted_model.save(self.params.model_path)
        return scripted_model
    
    def warmup(self, model):
        random_gen_img = torch.rand(self.params.TEST_BATCH, 3, self.params.width, self.params.height)
        random_gen_img =  random_gen_img.to(device)
        warmup_itr = 5
        for _ in range(warmup_itr):
            model(random_gen_img)

        return model

    def tvm_warmup(self, model):
        random_gen_img = torch.rand(self.params.TEST_BATCH, 3, self.params.width, self.params.height)
        #random_gen_img =  random_gen_img.to(device)
        warmup_itr = 5
        for _ in range(warmup_itr):
            model.set_input("input1", tvm.nd.array(random_gen_img.numpy()))
            model.run()

        return model
    
    def inference(self, model):
        inference_time = []
        results = pd.DataFrame()

        for iteration in range(self.params.test_iteration):
            for i, data in tqdm(enumerate(self.params.testloader)):
                input, labels = data 
                inputs, labels = data[0].to(device), data[1].to(device)
                start = time.time()
                out = model(inputs)
                end = time.time()
                inference_time.append((end-start)/self.params.TEST_BATCH)
            
            results[str(iteration) + "_time"] = inference_time

            
        dataframe_name = f'TVM_{self.params.tvm_status}_{self.params.model_name}_bch{str(self.params.TEST_BATCH)}.csv'

        results.to_csv(dataframe_name)
        print("Data is saved in ", dataframe_name)
        self.get_stats(dataframe_name)

    def evaluate(self, model):
        print("#### Evaluation Started ####")
        if self.params.tvm_enable:
            model.eval()

            # TODO Check set_input on gpu data
            # explore other options of PassContext
            # explore https://tvm.apache.org/docs/tutorial/autotvm_relay_x86.html
            # https://discuss.tvm.apache.org/t/whats-the-difference-between-build-and-create-executor-in-tvm-relay-build-module/1967/5
            # Use create_executor instead of build https://tvm.apache.org/docs/how_to/compile_models/from_keras.html

            #shape_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in  list(model.graph.inputs())[1:]]
            input_name = "input1"
            shape_list = [(input_name, (self.params.TEST_BATCH,3,self.params.height,self.params.width))]
            md, model_params = tvm.relay.frontend.pytorch.from_pytorch(model, shape_list, default_dtype="float32")
            target = tvm.target.Target("cuda", host="llvm")
            dev = tvm.cuda(0)

            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(md, target=target, params=model_params)
            
            model_graph = graph_executor.GraphModule(lib["default"](dev))

            model_graph = self.tvm_warmup(model_graph)

            inference_time = []

            for i, data in tqdm(enumerate(self.params.testloader)):
                input, labels = data 
                if input.shape != (self.params.TEST_BATCH,3, self.params.height,self.params.width):
                    print(f"Last batch of size {input.shape} ignored")
                    break

                model_graph.set_input(input_name, tvm.nd.array(input.numpy()))
                start_t = time.time()
                model_graph.run()
                end_t = time.time()
                # Get outputs
                tvm_output = model_graph.get_output(0)
                inference_time.append((end_t - start_t)/self.params.TEST_BATCH)
            
            results = pd.DataFrame()
            results["0_time"] = inference_time
            dataframe_name = f'TVM_{self.params.tvm_status}_{self.params.model_name}_bch{str(self.params.TEST_BATCH)}.csv'
            results.to_csv(dataframe_name)
            print("Data is saved in ", dataframe_name)
            self.get_stats(dataframe_name)

        else:
            model.eval()
            model = self.warmup(model)
            self.inference(model)


    def get_stats(self, csv_file):
        trace_d = pd.read_csv(csv_file)
        trace_d.drop(axis=1, inplace=True, index=0)
        time_columns = [col for col in trace_d.columns if 'time' in col]
        mean_time = trace_d[time_columns].mean().mean()
        print(f"Inference took {mean_time * 1000}ms")


if __name__ == "__main__":
    # python tvm_explore.py --create-model ownmodel --infe-batch 64
    # python tvm_explore.py --create-model ownmodel --infe-batch 64 --tvm
    parser = argparse.ArgumentParser()
    parser.add_argument('--tvm', action='store_true',
                        default=False, help="Set true to enable jit_compiler")
    parser.add_argument('--infe-batch', default=1,
                        help="Set the batch size used in inference", type=int)
    parser.add_argument('--only-train', action='store_true',
                        default=False, help="Train to save model")
    parser.add_argument('--create-model', default='resnet50',
                        help='set which model to use for inference')
    args = parser.parse_args()

    metadata = MetaData(model_name=args.create_model, test_batch=args.infe_batch, tvm_enable=args.tvm)
    modelpool = ModelPool(metadata)

    model_fn = {'resnet50': modelpool.create_resnet50,
                'resnet101': modelpool.create_resnet101,
                'vgg16': modelpool.create_vgg16,
                'ownmodel': modelpool.generate_model
                }

    print("Using Moldel ", metadata.model_name)    
    ground = PlayGround(metadata)

    if args.only_train:
        model = model_fn[args.create_model]()
        ground.train(model)
    else:
        model = torch.jit.load(metadata.model_path)        
        ground.evaluate(model)



