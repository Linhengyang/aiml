自行编写的依赖（包括但不限于模型、应用、架构、工具等代码），与业界流行依赖相比，总归是有一些参差的.
推荐处理办法: 自有依赖 总是和 自有依赖 交互（在 dl/rl 等环境中）, 业界依赖 总是和 业界依赖 交互（在 tf 等环境中）


kits 文件夹定义了一些开箱即用的工具，帮助对齐 自由依赖 与 业界依赖。比如 huggingface 文件夹中:
1. tokenizer_adapt 里修改 自有tokenizer, 以加载 huggingface/tokenizer 库标准的 tokenizer.json, 实现与 my tokenizer 对齐 huggingface/tokenizer
2. state_dict_adapt 里定义 state_dict 的 key_name 映射器, 以加载 huggingface/transformers 库标准的 model state_dict(weights), 实现 my network 对齐 huggingface/transformers

现在主要是 将业界标准 适配到 自有依赖上, 这样可以使用一些既有资源（一般都是huggingface标准）
未来可能要将 自有依赖 适配到 业界标准上, 这样可以将自己的东西分享出去