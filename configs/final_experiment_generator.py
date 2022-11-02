import json
import random
optimizers = ["SGD", "Adam"]
learning_rates = [1e-3, 1e-4]

small_deadlines = [1200 + i * 100 for i in range(12)]
big_deadlines = [2500 + i * 100 for i in range(12)]

big_epochs = [20, 30, 40]
small_epochs = [30, 40, 50]

print(random.choice(big_deadlines))

cifar_result = []
mnist_result = []

for optimizer in optimizers:
    for learning_rate in learning_rates:

        job = {
            "deadline": random.choice(big_deadlines),
            "classProbability": 0.25,
            "priorities": [
              {
                  "priority": 1,
                  "deadline": random.choice(big_deadlines),
                  "probability": 0.9
              },
                {
                  "priority": 0,
                  "deadline": random.choice(big_deadlines),
                  "probability": 0.1
              }
            ],
            "networkConfiguration": {
                "network": "Cifar10CNN",
                "lossFunction": "CrossEntropyLoss",
                "dataset": "cifar10"
            },
            "systemParameters": {
                "dataParallelism": 4,
                "configurations": {
                    "default": {
                        "cores": "2000m",
                        "memory": "2Gi"
                    }
                }
            },
            "hyperParameters": {
                "default": {
                    "totalEpochs": random.choice(big_epochs),
                    "batchSize": 128,
                    "testBatchSize": 128,
                    "learningRateDecay": 0.0002,
                    "optimizerConfig": {
                        "type": optimizer,
                        "learningRate": learning_rate
                    },
                    "schedulerConfig": {
                        "schedulerStepSize": 50,
                        "schedulerGamma": 0.5,
                        "minimumLearningRate": 1e-10
                    }
                },
                "configurations": {
                    "Master": None,
                    "Worker": None
                }
            },
            "learningParameters": {
                "cuda": False
            }
        }

        if optimizer == "Adam":
            job["hyperParameters"]["default"]["optimizerConfig"]["betas"] = [
                0.9, 0.999]
        else:
            job["hyperParameters"]["default"]["optimizerConfig"]["momentum"] = 0.9

        cifar_result.append(job)


# MNIST
for optimizer in optimizers:
    for learning_rate in learning_rates:

        job = {
            "deadline": random.choice(small_deadlines),
            "classProbability": 0.25,
            "priorities": [
              {
                  "priority": 1,
                  "deadline": random.choice(small_deadlines),
                  "probability": 0.9
              },
                {
                  "priority": 0,
                  "deadline": random.choice(small_deadlines),
                  "probability": 0.1
              }
            ],
            "networkConfiguration": {
                "network": "FashionMNISTCNN",
                "lossFunction": "CrossEntropyLoss",
                "dataset": "fashion-mnist"
            },
            "systemParameters": {
                "dataParallelism": 4,
                "configurations": {
                    "default": {
                        "cores": "2000m",
                        "memory": "2Gi"
                    }
                }
            },
            "hyperParameters": {
                "default": {
                    "totalEpochs": random.choice(small_epochs),
                    "batchSize": 128,
                    "testBatchSize": 128,
                    "learningRateDecay": 0.0002,
                    "optimizerConfig": {
                        "type": optimizer,
                        "learningRate": learning_rate
                    },
                    "schedulerConfig": {
                        "schedulerStepSize": 50,
                        "schedulerGamma": 0.5,
                        "minimumLearningRate": 1e-10
                    }
                },
                "configurations": {
                    "Master": None,
                    "Worker": None
                }
            },
            "learningParameters": {
                "cuda": False
            }
        }

        if optimizer == "Adam":
            job["hyperParameters"]["default"]["optimizerConfig"]["betas"] = [
                0.9, 0.999]
        else:
            job["hyperParameters"]["default"]["optimizerConfig"]["momentum"] = 0.9

        mnist_result.append(job)

assert len(mnist_result) == len(cifar_result)
assert len(mnist_result) == 4

with open("cifarjobs.json", "w") as f:
    f.write(json.dumps(cifar_result, indent=2))

with open("mnistjobs.json", "w") as f:
    f.write(json.dumps(mnist_result, indent=2))
