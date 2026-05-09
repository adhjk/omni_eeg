
## 1. 启动 Web 可视化GUI（必开，所有操作都在网页里）
```bash
streamlit run gui.py
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
使用数据训练的代码，如果需要修改路径就去config.yaml修改
oi-mi train-from-records --subject S001 --model eegnet