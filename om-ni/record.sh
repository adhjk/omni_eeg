# 临时：export PYTHONPATH="/mnt/dataset0/cst/omni/om-ni:$PYTHONPATH"
# TODO: 模块化task，让其方便通过一个指令从运动想象转到其他任务
# # 1. 测试模式（带cue+准确率计算）
# oi-mi run --test-mode --subject S001 --model riemann-mdm

# # 2. 纯实时在线解码
# oi-mi run --subject S001 --model riemann-mdm


# cat > .git/config << 'EOF'
# [core]
#         repositoryformatversion = 0
#         filemode = false
# [user]
#         name = adhjk
#         email = 1789377149@qq.com
# EOF



直接给你**当前整套项目、可直接复制运行**的三条核心测试命令，对应：校准 / 测试模式 / 实时解码，兼容你新的 `streamlit GUI + task_factory 多任务架构`

---

## 1. 启动 Web 可视化GUI（必开，所有操作都在网页里）
```bash
streamlit run web_gui.py
```
打开浏览器访问本地网页，所有参数、开始实验全部在页面点按钮操作。

---

## 2. 命令行执行 校准（原生motor默认任务）
```bash
oi-mi calibrate --new --subject S001
```

## 3. 命令行执行 测试模式
```bash
oi-mi run --test-mode --subject S001
```

## 4. 命令行执行 实时解码
```bash
oi-mi run --subject S001
```

---

## 切换为 visual 视觉任务 两种方式
### 方式1：改配置文件（永久生效）
打开 `config.yaml`
```yaml
task_mode: visual
```

### 方式2：后续代码可支持临时命令行传参（不用改配置）
后续扩展后可直接用：
```bash
oi-mi calibrate --new --subject S001 --task-mode visual
```

---

### 关键补充（对应你刚才的GUI疑问）
1. 无论命令行 / Streamlit网页GUI，都会自动读取 `task_mode`
   - 空/motor → 原左手/右手文字+箭头UI
   - visual → 你后续要做的：自定义提示语+EEG图片+音频
2. 代码里遗留的`LEFT/RIGHT`只影响**网页图标显示**，
   不改变marker、标签、模型、行为任务、数据存储。

需要我给你一条 **直接强制使用visual任务** 的完整测试命令吗？

