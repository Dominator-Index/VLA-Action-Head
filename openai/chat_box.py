import os
from openai import OpenAI

# 初始化OpenAI客户端（请替换为你自己的key和base_url）
client = OpenAI(
    api_key="sk-0fqDBukOpggdWA1H5d3a3971270042A7968505D9Dc2f2848",
    base_url="https://api.shubiaobiao.cn/v1"
)

def read_file(filepath: str, start_line: int = None, end_line: int = None) -> str:
    """
    读取指定文件的内容
    可选择读取指定的行区间[start_line, end_line]，行号从1开始
    如果不指定行号，则读取整个文件内容
    
    :param filepath: 文件完整路径
    :param start_line: 起始行（包含）
    :param end_line: 结束行（包含）
    :return: 读取的文本内容（字符串）
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")

    lines = []
    with open(filepath, 'r', encoding='utf-8') as f:
        if start_line is None and end_line is None:
            return f.read()
        else:
            for i, line in enumerate(f, start=1):
                if start_line is not None and i < start_line:
                    continue
                if end_line is not None and i > end_line:
                    break
                lines.append(line)
    return ''.join(lines)

def read_multiple_files(filepaths: list) -> str:
    """
    读取多个文件的内容，拼接成一个字符串
    
    :param filepaths: 文件路径列表
    :return: 所有文件内容拼接，带文件名标注
    """
    all_code = ""
    for filepath in filepaths:
        try:
            content = read_file(filepath)
            all_code += f"\n# File: {filepath}\n{content}\n"
        except Exception as e:
            print(f"读取文件 {filepath} 失败: {e}")
    return all_code.strip()

def read_folder_py_files(folder_path: str) -> str:
    """
    读取文件夹下所有 .py 文件内容，拼接成字符串
    
    :param folder_path: 目录路径
    :return: 所有.py文件内容拼接，带文件名标注
    """
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"不是目录: {folder_path}")

    all_code = ""
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.endswith(".py"):
                filepath = os.path.join(root, f)
                try:
                    content = read_file(filepath)
                    all_code += f"\n# File: {filepath}\n{content}\n"
                except Exception as e:
                    print(f"读取文件 {filepath} 失败: {e}")
    return all_code.strip()

def chat_with_code_context(code_context: str):
    """
    启动一个简单的多轮对话循环，模型基于code_context辅助回答
    
    :param code_context: 传给模型的上下文代码文本
    """
    messages = [
        {"role": "system", "content": "请用中文回答问题"},
        {"role": "system", "content": f"用户提供的Python代码上下文如下:\n{code_context}"},
    ]

    print("\n=== 多轮对话开始，输入 exit 退出 ===\n")
    while True:
        user_input = input("你说: ").strip()
        if user_input.lower() in ["exit", "quit", "退出"]:
            print("对话结束。")
            break

        messages.append({"role": "user", "content": user_input})

        try:
            response = client.chat.completions.create(
                model="claude-sonnet-4-20250514-thinking",
                stream=False,
                messages=messages
            )
            answer = response.choices[0].message.content
            print("助理:", answer)
            messages.append({"role": "assistant", "content": answer})

            # 保存对话历史（追加）
            with open("chat_history.md", "a", encoding="utf-8") as f:
                f.write(f"\n\n### User:\n```\n{user_input}\n```\n\n### Assistant:\n```\n{answer}\n```\n")

        except Exception as e:
            print(f"调用接口失败: {e}")

def main():
    print("请选择功能:")
    print("1. 读取单个文件（可指定行范围）")
    print("2. 读取多个文件（跨路径）")
    print("3. 读取文件夹内所有.py文件")
    print("4. 退出")

    choice = input("输入数字选择功能: ").strip()
    code_context = ""

    if choice == "1":
        filepath = input("请输入文件完整路径: ").strip()
        start_line_str = input("请输入起始行号（可不填，默认从头开始）: ").strip()
        end_line_str = input("请输入结束行号（可不填，默认读到文件末尾）: ").strip()

        start_line = int(start_line_str) if start_line_str.isdigit() else None
        end_line = int(end_line_str) if end_line_str.isdigit() else None

        try:
            code_context = read_file(filepath, start_line, end_line)
            print(f"已读取文件内容（长度 {len(code_context)} 字符）")
        except Exception as e:
            print(f"读取文件失败: {e}")
            return

    elif choice == "2":
        print("请输入多个文件路径（换行分隔），输入空行结束:")
        filepaths = []
        while True:
            line = input()
            if line.strip() == "":
                break
            filepaths.append(line.strip())
        if not filepaths:
            print("未输入任何文件路径，退出。")
            return
        code_context = read_multiple_files(filepaths)
        print(f"已读取 {len(filepaths)} 个文件，总长度约 {len(code_context)} 字符")

    elif choice == "3":
        folder_path = input("请输入文件夹路径: ").strip()
        try:
            code_context = read_folder_py_files(folder_path)
            print(f"已读取文件夹下所有.py文件内容，总长度约 {len(code_context)} 字符")
        except Exception as e:
            print(f"读取文件夹失败: {e}")
            return

    elif choice == "4":
        print("退出程序")
        return

    else:
        print("无效选择，退出程序")
        return

    # 启动多轮对话
    chat_with_code_context(code_context)


if __name__ == "__main__":
    main()
