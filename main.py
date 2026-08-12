"""Canonical root entrypoint for MyCodeAgent."""

# 数据流起点：python main.py → cli.main()
# cli.main() 负责两件事：1) 组装所有依赖（build_runtime）2) 驱动交互循环
from app.cli import main


if __name__ == "__main__":
    main()
