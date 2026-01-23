自行编写的依赖（包括但不限于模型、应用、架构、工具等代码），与业界流行依赖相比，总归是有一些参差的.
推荐处理办法: 自有依赖 总是和 自有依赖 交互（在 dl/rl 等环境中）, 业界依赖 总是和 业界依赖 交互（在 tf 等环境中）


kits 文件夹定义了一些开箱即用的 移植工具，帮助对齐 custom 与 popular。比如 huggingface 文件夹中:
1. tokenizer_adapt 里修改 自有tokenizer, 以加载 huggingface/tokenizer 库标准的 tokenizer.json, 实现 custom tokenizer 对齐 huggingface/tokenizer
2. state_dict_adapt 里修改 huggingface/transformers 库标准的 model state_dict(weights), 以适应并载入 custom network
这样通过移植 huggingface 到 custom, 使得以 custom 为底层, 运行 huggingface 标准的 tokenizer / model resource


当前主要是 将 huggingface 移植成 custom, 这样可以使用社区的既有 huggingface 标准的资源
未来可能要将 custom 移植成 huggingface, 这样可以将自己的东西以 huggingface 标准分享出去