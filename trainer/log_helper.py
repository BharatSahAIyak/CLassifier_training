import pandas as pd
from matplotlib import pyplot as plt

def separate_train_eval_logs(logs):
    # take only column starting with train as prefix and filter epocch wise
    train_logs = logs[[col for col in logs.columns if col.startswith(("train", "epoch"))]]
    eval_logs = logs[[col for col in logs.columns if col.startswith(("eval", "epoch"))]]

    # filter even numbered rows for train and odd numbered rows for eval as per row id
    train_logs = train_logs.iloc[[i for i in range(0, len(train_logs), 2)]]
    eval_logs = eval_logs.iloc[[i for i in range(1, len(eval_logs), 2)]]

    return train_logs, eval_logs

def save_log_graphs(train_logs, eval_logs):
    # plot train and eval loss
    plt.plot(train_logs['epoch'], train_logs['train_loss'], label='train_loss')
    plt.plot(eval_logs['epoch'], eval_logs['eval_loss'], label='eval_loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Eval Loss')
    plt.savefig('../logs/train_eval_loss.png')
    plt.close()

    # plot train and eval f1
    plt.plot(train_logs['epoch'], train_logs['train_f1'], label='train_f1')
    plt.plot(eval_logs['epoch'], eval_logs['eval_f1'], label='eval_f1')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.title('Train and Eval F1')
    plt.savefig('../logs/train_eval_f1.png')
    plt.close()

    # plot train and eval precision
    plt.plot(train_logs['epoch'], train_logs['train_precision'], label='train_precision')
    plt.plot(eval_logs['epoch'], eval_logs['eval_precision'], label='eval_precision')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Train and Eval Precision')
    plt.savefig('../logs/train_eval_precision.png')
    plt.close()

    # plot train and eval recall
    plt.plot(train_logs['epoch'], train_logs['train_recall'], label='train_recall')
    plt.plot(eval_logs['epoch'], eval_logs['eval_recall'], label='eval_recall')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Train and Eval Recall')
    plt.savefig('../logs/train_eval_recall.png')
    plt.close()

    # plot train and eval accuracy
    plt.plot(train_logs['epoch'], train_logs['train_accuracy_score'], label='train_accuracy')
    plt.plot(eval_logs['epoch'], eval_logs['eval_accuracy_score'], label='eval_accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train and Eval Accuracy')
    plt.savefig('../logs/train_eval_accuracy.png')


def save_logs(logs):
    logs = [log for log in logs if log['epoch'].is_integer()][:-1]
    logs = pd.DataFrame(logs)

    train_logs, eval_logs = separate_train_eval_logs(logs)
    train_logs.to_csv('../logs/train_logs.csv', index=False)
    eval_logs.to_csv('../logs/eval_logs.csv', index=False)

    save_log_graphs(train_logs, eval_logs)