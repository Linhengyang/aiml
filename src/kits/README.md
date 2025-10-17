自行编写的依赖（包括但不限于模型、应用、架构、工具等代码），与业界流行依赖相比，总归是有一些参差的.
推荐处理办法: 自有依赖 总是和 自有依赖 交互（在 dl/rl 等环境中）, 业界依赖 总是和 业界依赖 交互（在 tf 等环境中）


kits 文件夹定义了一些开箱即用的工具，帮助对齐 自由依赖 与 业界依赖。比如:
1. tokenizer_kit 里修改 自有tokenizer, 以加载 huggingface/tokenizer 库标准的 tokenizer.json, 实现与 my tokenizer 对齐 huggingface/tokenizer
2. load_kit 里定义 state_dict 的 key_name 映射器, 以加载 huggingface/transformers 库标准的 model state_dict(weights), 实现 my network 对齐 huggingface/transformers