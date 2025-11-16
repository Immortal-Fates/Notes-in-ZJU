# Main Takeaway

经常需要刷机，因此在此记录一下刷机全流程

<!--more-->


# 刷机装系统

1. 下载iso
2. 分区压缩卷
3. U盘使用rufus登制作启动盘
4. 目标电脑进BIOS关闭安全模式，选择U盘启动然后install



# 新电脑环境配置

> 事先声明：平时开发主要用IDE，配置这些有的没的主要是闲的，偶尔懒得开vscode会用，lua不太会

## ubuntu

### 系统安装

- 换源及更新

  > 如果是从别人软件站下的可能已经换好了

  1. 在软件和更新中（在zju推荐直接使用浙大源）

  2. ```
     sudo apt update & upgrade -y
     ```

- 系统设置

  - 设置更新频率：在软件和更新中，在更新中设置
  - 附加驱动：完善硬件驱动

- 关闭 sudo 密码

  为了避免每次使用 sudo 命令时都输入密码，我们可以将密码关闭。操作方法：

  1. 终端输入命令`sudo visudo`，打开 visudo；
  2. 找到 `%sudo ALL=(ALL:ALL) ALL` 这一行修改为`%sudo ALL=(ALL:ALL) NOPASSWD:ALL`

  > 有安全风险，请谨慎使用
  >
  > 可能使用nano打开，可以先装一个vim
  >
  > ```
  > sudo apt install vim            # if you don’t have vim yet
  > sudo update-alternatives --config editor
  > ```

- 安装GCC，GNU等

  ```
  sudo apt install build-essential
  ```

- 高分屏适配：settings->Displays开启HiDPI支持

### 一键配置

- git

  ```
  sudo apt install git
  git config --global user.email ""
  git config --global user.name ""
  ```

- chezmoi



### Terminal and Shell

**Terminal 是“显示/输入的窗口（终端模拟器）仪表盘+方向盘”，Shell 是“命令解释器（你跟系统对话的大脑）引擎”。** 它们常一起出现，但不是一回事。

```
键盘/显示器 ←→ Terminal(终端/PTY) ←→ Shell(bash/zsh/...) ←→ 各种程序与内核
```

- terminal: Wezterm（快Rust实现，跟风）

  - install: [Linux - Wez's Terminal Emulator](https://wezterm.org/install/linux.html)

    ```
    # 设置为默认终端
    sudo update-alternatives --config x-terminal-emulator
    ```

- shell: Zsh

  - 安装zsh

    ```
    # 安装 zsh
    apt install zsh
    
    # 将 zsh 设置为系统默认 shell
    sudo chsh -s $(which zsh)
    ```

  - zsh theme: starship(rust so quick)

    ```
    curl -sS https://starship.rs/install.sh | sh
    ```

    if can`t, install Homebrew, then 

    ```
    brew install starship
    ```

    config:

    ```
    mkdir -p ~/.config && touch ~/.config/starship.toml
    starship preset pastel-powerline -o ~/.config/starship.toml
    ```

  - 使用oh-my-zsh方便配置我的zsh，zsh插件安装

    ```
    # autojump快速切换目录
    apt install autojump
    # 使用
    j Document/
    
    zsh-autosuggestions：命令行命令键入时的历史命令建议插件
    git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
    
    zsh-syntax-highlighting：命令行语法高亮插件
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
    
    zsh-sudo: 两次Esc加入sudo
    git clone git@github.com:none9632/zsh-sudo.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-sudo
    
    zsh-vi-mode: 在zsh中使用vim
    git clone https://github.com/jeffreytse/zsh-vi-mode \
      $ZSH_CUSTOM/plugins/zsh-vi-mode
    ```

    安装好了配置在`~/.zshrc`

    ```
    # 打开 ~/.zshrc 文件，找到如下这行配置代码，在后面追加插件名
    plugins=(其他插件名 autojump sudo zsh-vi-mode zsh-autosuggestions zsh-syntax-highlighting)
    ```

    > sudo 和vi-mode有冲突，我宁愿放弃sudo

- font:

  ```
  # 普通版
  sudo apt install fonts-jetbrains-mono -y
  ```

  ```
  # Nerd Font
  mkdir -p ~/.local/share/fonts
  cd ~/.local/share/fonts
  
  # 下载最新版 JetBrainsMono Nerd Font
  wget https://github.com/ryanoasis/nerd-fonts/releases/latest/download/JetBrainsMono.zip
  
  # 解压并更新缓存
  unzip -o JetBrainsMono.zip
  fc-cache -fv
  
  ```

- CMatrix黑客帝国

  ```
  # 安装
  sudo apt install cmatrix
  
  # 运行（加上 -lba 参数看起来更像电影，加上 -ol 参数起来更像 Win/Mac 的屏保）
  cmatrix
  ```

- neovim:

  - installation: 上github下载新版，CIL下载的是很老的版本


### 软件安装

- 搭梯子：clash verge (on github)，然后来glados找自己的配置文件导入即可

- typora

  ```
  # add Typora's key
  sudo mkdir -p /etc/apt/keyrings
  curl -fsSL https://typoraio.cn/linux/typora.gpg | sudo tee /etc/apt/keyrings/typora.gpg > /dev/null
  # add Typora's repository securely
  echo "deb [signed-by=/etc/apt/keyrings/typora.gpg] https://typoraio.cn/linux ./" | sudo tee /etc/apt/sources.list.d/typora.list
  sudo apt update
  # install typora
  sudo apt install typora
  ```

- vscode/cursor



### 桌面美化

- picom 设置软件透明化

TODO 





TODO

> 这些全都会给你下载到C盘，不想这样就自己上官网手动下载

- chezmoi下载

  ```
  winget install chezmoi
  ```

- wezterm

  ```
  winget install wez.wezterm
  ```

  设置为默认打开终端

- 下载字体`JetBrainsMono Nerd Font`

  安装到系统中，同时也可以给vscode使用

- 安装msys2——[msys2 | 镜像站使用帮助 | 清华大学开源软件镜像站 | Tsinghua Open Source Mirror](https://mirrors.tuna.tsinghua.edu.cn/help/msys2/)

  - pacman配置

    ```
    sed -i "s#https\?://mirror.msys2.org/#https://mirrors.tuna.tsinghua.edu.cn/msys2/#g" /etc/pacman.d/mirrorlist*
    ```

    



## MacOS

# CheetSheet

- wezterm

  ```
  alt + hjkl 不同分屏之间移动
  ctrl+shift + o/e 不同分屏
  alt+enter 全屏屏幕
  ```

  Ubuntu / GNOME 桌面自带最小化功能：

  - **`Super + H`** → 隐藏（最小化）当前窗口











# References

- [Windows11 + Linux (Ubuntu22.04) 双系统最简安装详细避坑版_win11安装linux双系统-CSDN博客](https://blog.csdn.net/2401_84064328/article/details/137232169)
- [(39 封私信 / 80 条消息) 写给工程师的 Ubuntu 20.04 最佳配置指南 - 知乎](https://zhuanlan.zhihu.com/p/139305626)
- [2024年最好用的12款 Linux Terminal Emulator (终端模拟器)精心挑选了2024年最值得使用的 - 掘金](https://juejin.cn/post/7317832600810160191)

- [(39 封私信 / 80 条消息) 终端配置一把梭：wezterm、neovim、zsh、鼠须管... - 知乎](https://zhuanlan.zhihu.com/p/700501866)
