# 我的个人博客

这是一个用 Hugo + Archie 主题搭建的个人博客，同时可以作为 Obsidian vault 使用。

## 写文章

用 Obsidian 打开这个文件夹：

```text
/Users/mio/Documents/Codex/2026-06-24/ni-xu/blog
```

文章放在：

```text
content/posts
```

每篇文章顶部的 `draft: false` 表示会发布，`draft: true` 表示草稿。

## 本地预览

```bash
hugo server --buildDrafts
```

## 生成静态网站

```bash
hugo
```

生成结果会在 `public` 目录里。

## 免费发布

推荐使用 GitHub Pages。仓库推到 GitHub 后，GitHub Actions 会自动构建并发布。

发布前需要把 `hugo.yaml` 里的 `baseURL` 改成真实地址，例如：

```yaml
baseURL: https://你的用户名.github.io/仓库名/
```
