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
  - Very important: the trailing slash
    - With trailing slash `rsync -av ./project/ user@server:~/project/`

      Meaning: “copy the _contents_ of `./project/` into `~/project/`”.

    - Without trailing slash `rsync -av ./project user@server:~/project/`

      Meaning: “copy the directory `project` itself (as a folder) into `~/project/` → you get `~/project/project/`”.

  - Keeping folders exactly in sync: `--delete`

    `--delete`: delete files on the target that don’t exist on the source.

  - Excluding files/directories: `--exclude`

## SSH Key Pair Authentication

### Generate a Key Pair (local machine)

```
ssh-keygen -t ed25519
```

Files created:

- Private key: `~/.ssh/id_ed25519`
- Public key: `~/.ssh/id_ed25519.pub`

### Copy Public Key to Remote Server

```
ssh-copy-id username@remote_ip
```

After this, you can log in without password.

If you set a alias in the ssh/config, just `ssh-copy-id $(alias)`

## Port Forwarding

In many scenarios you will run into software that listens to specific ports in the machine.

`localhost:PORT` or `127.0.0.1:PORT`

But what do you do with a remote server that does not have its ports directly available through the network/internet? -- _port forwarding_

- Local Port Forwarding

  ![Local Port Forwarding](https://missing.csail.mit.edu/static/media/images/local-port-forwarding.png)

- Remote Port Forwarding

  ![Remote Port Forwarding](https://missing.csail.mit.edu/static/media/images/remote-port-forwarding.png)

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
  1. (slow)Mount the project file on your local folder and edit it like the other local files.
  2. (highly recommended)Use rsync cmd to sync the project
  3. (with collaborators)Use git to sync.
