# utils/logger.py

class TrainLogger:
    def __init__(self):
        self.epochs = []
        self.train_loss = []
        self.train_acc = []
        self.test_acc = []
        self.rest_acc = {}   # epoch → rest accuracy

    def log_train(self, epoch, loss, train_acc, test_acc):
        self.epochs.append(epoch)
        self.train_loss.append(loss)
        self.train_acc.append(train_acc)
        self.test_acc.append(test_acc)

    def log_rest(self, epoch, rest_accuracy):
        self.rest_acc[epoch] = rest_accuracy
