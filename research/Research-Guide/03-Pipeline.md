# Main Takeaway

介绍一下我的科研pipeline

> 完善ing

<!--more-->

# 文献管理及阅读

> 【我是如何快速阅读和整理文献】<https://www.bilibili.com/video/BV1nA41157y4?vd_source=93bb338120537438ee9180881deab9c1>

1. 在python中安装autoliter[GitHub - wilmerwang/autoLiterature: autoLiterature](https://github.com/WilmerWang/autoLiterature)

2. 在Google Scholar中搜索文献，看摘要筛选一篇

   - 在浏览器中下载easyScholar插件方便获取具体的期刊信息

   - 使用[Litmaps](https://app.litmaps.com/)根据引用查找文献

3. 将筛选得到的文献的DOI或者arXiv ID按照如下格式放在md中

   | 仅下载元数据   | `- {paper_id}`   |
   | -------------- | ---------------- |
   | 下载元数据+PDF | `- {{paper_id}}` |

4. 使用autoliter下载元数据和PDF

   ```
   autoliter -i ./**.md -o PDF_folder
   ```

   > - 这时常会下载不成功需要自行下载PDF放在本地并建立链接
   > - 如果是因为GBK不行需要添加一句`set PYTHONUTF8=1`
   > - 不开梯子有时能够成功抓取，但是写入的链接是多了../ ？

5. 直接在vscode中读文献，边看PDF，边写note，还能用copilot

6. 如果文献多了还可以使用mermaid建立框图

# References

- 【我是如何快速阅读和整理文献】<https://www.bilibili.com/video/BV1nA41157y4?vd_source=93bb338120537438ee9180881deab9c1>
- [(36 封私信 / 80 条消息) Google Scholar终极使用指南 - 知乎](https://zhuanlan.zhihu.com/p/107911957?theme=dark)
