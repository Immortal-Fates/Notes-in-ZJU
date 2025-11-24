# tmux

When I am running a deep learning task on a remote server, tmux and screen are suggested for me to use. So it`s time to dive into the tmux.

## What

First we should know the difference showed below:

- **Terminal emulator**: GUI app that shows a terminal window
  - Examples: WezTerm, iTerm2, Alacritty, Windows Terminal
- **Terminal multiplexer**: Manages panes/windows/sessions *inside* a terminal
  - Examples: tmux, GNU screen
- **Shell**: Runs commands
  - Examples: zsh, bash, fish

**tmux = terminal multiplexer (runs on the machine where your shell runs)**

- Runs on the **remote or local host** (e.g., your GPU server).
- Provides:
  - Persistent sessions (`tmux attach` after disconnect)
  - Windows and panes (splits)
  - Session sharing (multiple clients attach)
- Keeps things alive when:
  - Your SSH connection drops
  - Your local terminal (WezTerm, etc.) crashes

> Tips: tmux does not draw a GUI; it just controls text inside the terminal.

## Why

- **Persistent Sessions** (Survive Disconnects)

  tmux keeps your session running even if:

  - You close the terminal,

  - Your SSH connection drops,

    > Notes: Why if I don`t use tmux, when my ssh drops, the running process will die:
    >
    > - When the terminal disappears, the OS sends a SIGHUP (hangup) to your processes
    >
    > - Most deep learning training scripts are *not* daemonized because:
    >
    >   - You want logs printed to stdout
    >
    >   - You want a terminal
    >
    >   - You want user interaction if needed

  - You restart your local terminal window.

  This is essential for long-running training jobs, remote server work, and debugging.

- Session Multiplexing & Sharing

  tmux lets multiple clients attach to the same session.

  Use cases:

  - Pair programming,
  - Monitoring a remote experiment,
  - Teaching or debugging with others.

  Shells cannot share or multiplex views.

- Scrollback History That Survives Clear or Program Output

  tmux maintains its own scrollback buffer, independent from the terminal emulator.

  This is useful when:

  - Python training logs exceed 10,000 lines,
  - You need to scroll after running a noisy program like `dmesg` or a training log.

- screen is old and tmux is new.

## How(Tmux Cheat Sheet)

### 1. Session Management

| Action            | Command / Key                | Explanation                                   |
| ----------------- | ---------------------------- | --------------------------------------------- |
| Create session    | `tmux new -s train`          | Start a new tmux session named "train".       |
| List sessions     | `tmux ls`                    | Show all existing sessions.                   |
| Attach to session | `tmux attach -t train`       | Re-enter the session after disconnecting.     |
| Detach session    | `Ctrl-b d`                   | Leave the session but keep processes running. |
| Kill session      | `tmux kill-session -t train` | Terminate the session and all tasks in it.    |

---

### 2. Window (Tab) Management

| Action                    | Key        | Explanation                              |
| ------------------------- | ---------- | ---------------------------------------- |
| Create new window         | `Ctrl-b c` | Add a new window (like a tab).           |
| Switch to next window     | `Ctrl-b n` | Move to the next window.                 |
| Switch to previous window | `Ctrl-b p` | Move to the previous window.             |
| Switch to window number N | `Ctrl-b N` | Jump directly to that window.            |
| Rename current window     | `Ctrl-b ,` | Helps organize tasks (train, logs, gpu). |

---

### 3. Pane (Split) Management

| Action                    | Key                        | Explanation                |
| ------------------------- | -------------------------- | -------------------------- |
| Split window vertically   | `Ctrl-b %`                 | Left & right split.        |
| Split window horizontally | `Ctrl-b "`                 | Top & bottom split.        |
| Move between panes        | `Ctrl-b` + arrow keys      | Navigate between splits.   |
| Resize pane               | `Ctrl-b Ctrl` + arrow keys | Adjust pane size.          |
| Close current pane        | `Ctrl-b x`                 | Kill only the active pane. |

---

### 4. Useful Commands for Deep Learning

| Task                  | Command                                 | Explanation                           |
| --------------------- | --------------------------------------- | ------------------------------------- |
| GPU monitoring        | `watch -n 1 nvidia-smi`                 | Real-time GPU usage.                  |
| CPU/memory monitoring | `htop`                                  | Interactive system usage.             |
| Tail logs             | `tail -f log.txt`                       | Monitor logs live.                    |
| Run TensorBoard       | `tensorboard --logdir runs --port 6006` | Launch TensorBoard in its own window. |

---

### 5. Recovery Workflow (After Disconnect)

| Step          | Command           | Explanation                                  |
| ------------- | ----------------- | -------------------------------------------- |
| SSH to server | `ssh user@server` | Connect to the remote machine.               |
| List sessions | `tmux ls`         | Check currently running sessions.            |
| Reattach      | `tmux a -t train` | Restore your training environment instantly. |

---

### 6. Recommended `.tmux.conf` Settings

| Setting             | Line                          | Explanation                   |
| ------------------- | ----------------------------- | ----------------------------- |
| Enable mouse        | `set -g mouse on`             | Scroll and select with mouse. |
| Long scrollback     | `set -g history-limit 100000` | Keep long logs in buffer.     |
| Vim-style copy mode | `setw -g mode-keys vi`        | Convenient navigation.        |

---

### 7. Minimal Daily Workflow

| Task             | Command                                 | Explanation                           |
| ---------------- | --------------------------------------- | ------------------------------------- |
| Start training   | `tmux new -s train` → `python train.py` | Begin training safely inside tmux.    |
| Detach           | `Ctrl-b d`                              | Training continues even if SSH drops. |
| Resume           | `tmux a -t train`                       | Reconnect anytime.                    |
| Open log window  | `Ctrl-b c` → `tail -f log.txt`          | Monitor logs separately.              |
| Open GPU monitor | Split pane → `watch -n 1 nvidia-smi`    | Real-time GPU usage.                  |