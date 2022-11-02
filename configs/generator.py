from pyDOE2 import fracfact
import json
generators = "a b c d e ab bc ac ad ae bd be cd ce de abc abd abe bcd bce acd cde ace ade bde abcd bcde acde abde abce"

table = fracfact(generators)

print(table.shape)


parameters = [
    ["CNN", "ResNet"],
    [["fashion-mnist", "FashionMNIST"],  ["cifar10", "Cifar10"]],
    ["SGD", "Adam"],
    [40, 80],
    [1e-3, 1e-4]
]


experiments_all = [
    [1, 0, 0, 0, 0],  # A
    [0, 1, 0, 0, 0],  # B
    [0, 0, 1, 0, 0],  # C
    [0, 0, 0, 1, 0],  # D
    [0, 0, 0, 0, 1],  # E

    [1, 1, 0, 0, 0],  # AB
    [0, 1, 1, 0, 0],  # BC
    [1, 0, 1, 0, 0],  # AC
    [1, 0, 0, 1, 0],  # AD
    [1, 0, 0, 0, 1],  # AE

    [0, 1, 0, 1, 0],  # BD
    [0, 1, 0, 0, 1],  # BE
    [0, 0, 1, 1, 0],  # CD
    [0, 0, 1, 0, 1],  # CE
    [0, 0, 0, 1, 1],  # DE

    [1, 1, 1, 0, 0],  # ABC
    [1, 1, 0, 1, 0],  # ABD
    [1, 1, 0, 0, 1],  # ABE
    [0, 1, 1, 1, 0],  # BCD
    [0, 1, 1, 0, 1],  # BCE
    [1, 0, 1, 1, 0],  # ACD

    [0, 0, 1, 1, 1],  # CDE
    [1, 0, 1, 0, 1],  # ACE
    [1, 0, 0, 1, 1],  # ADE
    [0, 1, 0, 1, 1],  # BDE
    [1, 1, 1, 1, 0],  # ABCD

    [0, 1, 1, 1, 1],  # BCDE
    [1, 0, 1, 1, 1],  # ACDE
    [1, 1, 0, 1, 1],  # ABDE
    [1, 1, 1, 0, 1],  # ABCE
    [1, 1, 1, 1, 1],  # ABCDE
    [0, 0, 0, 0, 0],  # NONE
]

experiments_lorenzo = []
experiments_chris = []

count = 0
for i in experiments_all:
    if i[0] == 0:
        if count % 2 == 0:
            experiments_lorenzo.append(i)
        else:
            experiments_chris.append(i)
        count += 1

DATA_PARALLELISM = 4


def generate(experiments):
    tasks = {
        "trainTasks": []
    }

    seeds = [69, 420, 1337]

    count = -1
    for seed in seeds:
        for exp in experiments:
            count += 1
            network = parameters[0][exp[0]]
            dataset = parameters[1][exp[1]]
            optimizer = parameters[2][exp[2]]
            epochs = parameters[3][exp[3]]
            lr = parameters[4][exp[4]]

            task = {
                "type": "distributed",
                "jobClassParameters": [
                    {
                        "networkConfiguration": {
                            "network": f"{dataset[1]}{network}",
                            "lossFunction": "CrossEntropyLoss",
                            "dataset": dataset[0]
                        },
                        "systemParameters": {
                            "dataParallelism": DATA_PARALLELISM,
                            "configurations": {
                                "default": {
                                    "cores": "2000m",
                                    "memory": "2Gi"
                                }
                            }
                        },
                        "hyperParameters": {
                            "default": {
                                "totalEpochs": epochs,
                                "batchSize": 128,
                                "testBatchSize": 128,
                                "learningRateDecay": 0.0002,
                                "optimizerConfig": {
                                    "type": optimizer,
                                    "learningRate": lr,
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
                            "totalEpochs": epochs,
                            "rounds": epochs,
                            "epochsPerRound": 1,
                            "cuda": False,
                            "clientsPerRound": DATA_PARALLELISM,
                            "dataSampler": {
                                "type": "uniform",
                                "qValue": 0.07,
                                "seed": seed,
                                "shuffle": True
                            },
                            "aggregation": "FedAvg"
                        }
                    }
                ]
            }

            if optimizer == "Adam":
                task["jobClassParameters"][0]["hyperParameters"]["default"]["optimizerConfig"]["betas"] = [
                    0.9, 0.999]
            else:
                task["jobClassParameters"][0]["hyperParameters"]["default"]["optimizerConfig"]["momentum"] = 0.9

            tasks["trainTasks"].append(task)
    return tasks


lorenzo_tasks = generate(experiments_lorenzo)
chris_tasks = generate(experiments_chris[:])

print(len(chris_tasks['trainTasks']))

with open("lorenzo_tasks.json", "w") as f:
    f.write(json.dumps(lorenzo_tasks, indent=2))


with open("chris_tasks_remainder.json", "w") as f:
    f.write(json.dumps(chris_tasks, indent=2))
