# SSH

SSH (Secure Shell) is a protocol for secure remote login and command execution. It encrypts traffic, supports key-based authentication, and is widely used to access remote Linux servers.

[TOC]

## Basic Operation

- login in: `ssh username@remote_ip`
- Specify Port: `ssh -p 2222 user@host`
- Executing a Command Without Opening a Shell: `ssh user@server "ls -al /home/user"`

## File Transfer with SSH

- Upload a file: `scp local.txt user@server:/path/`

- Download a file: `scp user@server:/path/file.txt .`

- rsync (fast, incremental sync): `rsync -avz ./project/ user@server:/home/user/project/`

## SSH Key Pair Authentication

### Generate a Key Pair (local machine)

```
ssh-keygen -t ed25519
```

Files created:

- Private key: `~/.ssh/id_ed25519`
- Public key:  `~/.ssh/id_ed25519.pub`

### Copy Public Key to Remote Server

```
ssh-copy-id username@remote_ip
```

After this, you can log in without password.

## SSHFS (mount remote directory locally)

- install: `sudo apt install sshfs`

- Mount: 

  ```
  mkdir ~/remote
  sshfs user@server:/path ~/remote
  ```

- Unmount: `fusermount -u ~/remote`

## Lessons Learned

- If you want to use AI to assist you and running on your local computer. There are three ways
  1. Mount the project file on your local folder and edit it like the other local files.
  2. Use rsync cmd to sync the project
  3. Use git to sync.