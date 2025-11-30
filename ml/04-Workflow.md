# Workflow

show workflow when training on local or remote server.

## Useful Cmds

| Task                  | Command                                 | Explanation                           |
| --------------------- | --------------------------------------- | ------------------------------------- |
| GPU monitoring        | `watch -n 1 nvidia-smi`                 | Real-time GPU usage.                  |
| CPU/memory monitoring | `htop/top`                              | Interactive system usage.             |
| Tail logs             | `tail -f log.txt`                       | Monitor logs live.                    |
| Run TensorBoard       | `tensorboard --logdir runs --port 6006` | Launch TensorBoard in its own window. |

- [check nn architecture](https://netron.app/)

## Remote Server

- first connect

  Generate a Key Pair(local machine)

  ```
  ssh-keygen -t rsa(whatever, you may already have)
  ```

  Copy Public Key to Remote Server

  ```
  ssh-copy-id username@remote_ip
  ```

  After this, you can log in without password.

- normal

| Task                     | Command                                                      | Explanation                               |
| ------------------------ | ------------------------------------------------------------ | ----------------------------------------- |
| ssh connect              | `ssh user@ip`                                                | remote connect                            |
| Sync code                | `git` or `rsync -avz --delete ~/ws/remote-project/ root@connect.nmb2.seetacloud.com:/root/Test-Example/` | Sync the code                             |
| Start training           | `tmux new -s train` → `python train.py`                      | Begin training safely inside tmux.        |
| Detach                   | `Ctrl-b d`                                                   | Training continues even if SSH drops.     |
| Resume                   | `tmux a -t train`                                            | Reconnect anytime.                        |
| Open tb forward the port | On the server: `tensorboard --logdir experiments --port 6006` | Add a TB writer and then forward the port |
| Local open TB            | `ssh -L 6006:localhost:6006 user@server`<br />[http://localhost:6006](http://localhost:6006/) | Check TB on local machine                 |
| Open log window          | `Ctrl-b c` → `tail -f log.txt`                               | Monitor logs separately.                  |
| Open GPU monitor         | Split pane → `watch -n 1 nvidia-smi`                         | Real-time GPU usage.                      |

## Hyperparameter tuning

use **Optuna** for hyperparameter tuning
