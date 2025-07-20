import os
import json
import re
import fitz  # PyMuPDF
from openai import OpenAI
from typing import Generator, Optional, List, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path

class PaperAnalyzer:
    """
    一个用于分析学术论文PDF的类。
    它封装了从PDF提取文本、调用大模型进行分析以及处理多个任务的功能。
    """

    def __init__(self, api_key: str, base_url: str = "https://api.siliconflow.cn/v1",
                 model: str = "Qwen/Qwen2.5-72B-Instruct"):
        """
        初始化分析器。

        :param api_key: 用于访问大模型API的密钥。
        :param base_url: API服务的URL。
        :param model: 要使用的模型名称。
        """
        if not api_key:
            raise ValueError("API key cannot be empty.")
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _extract_text_from_pdf(self, file_path: str) -> Optional[str]:
        """
        [内部方法] 从PDF文件中提取全部文本内容。
        """
        if not os.path.exists(file_path):
            print(f"[错误] 文件未找到: {file_path}")
            return None
        try:
            doc = fitz.open(file_path)
            full_text = "\n".join([page.get_text() for page in doc])
            doc.close()
            return full_text
        except Exception as e:
            print(f"[错误] 提取PDF文本时发生异常: {e}")
            return None

    def _stream_chat_response(self, full_prompt: str) -> Generator[str, None, None]:
        """
        [内部方法] 向大模型发送Prompt并获取流式响应。
        """
        messages = [{'role': 'user', 'content': full_prompt}]
        try:
            response_stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True
            )
            for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            error_message = f"\n[API请求出错]: {e}"
            print(error_message)
            yield error_message

    def _process_single_task(self, task_config: Dict[str, Any], full_text: str) -> str:
        """
        [内部方法] 处理一个独立的分析任务。
        """
        task_name = task_config["name"]
        instruction = task_config["instruction"]
        not_found_indicator = task_config.get("not_found_indicator", "TASK_CONTENT_NOT_FOUND")

        print(f"任务 '{task_name}' 正在分析全文...")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=500
        )
        chunks = text_splitter.split_text(full_text)
        print(f"全文被切分为 {len(chunks)} 块进行处理。")

        task_responses = []
        for i, chunk in enumerate(chunks):
            print(f"  -> 正在处理 '{task_name}' 的第 {i + 1}/{len(chunks)} 块...")
            prompt = f"{instruction}\n\n--- 待分析文本块 ---\n{chunk}\n--- 文本块结束 ---"

            response_generator = self._stream_chat_response(prompt)
            response_content = "".join(list(response_generator))

            if not_found_indicator not in response_content:
                task_responses.append(response_content)

        if not task_responses:
            return "在全文的所有文本块中均未找到或生成相关内容。"

        return "\n\n".join(task_responses).strip()

    def analyze_paper(self, pdf_path: str, tasks: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        [公共方法] 分析一篇指定的PDF论文，执行所有定义的任务。

        :param pdf_path: 要分析的PDF文件的路径。
        :param tasks: 一个包含多个任务配置字典的列表。
        :return: 一个包含所有任务结果的字典。
        """
        print(f"===== 开始分析论文: {pdf_path} =====")
        full_text = self._extract_text_from_pdf(pdf_path)
        if not full_text:
            print("因文本提取失败，分析终止。")
            return {}

        final_results = {}
        for task in tasks:
            print(f"\n--- 开始执行任务: {task['name']} ---")
            result = self._process_single_task(task, full_text)
            final_results[task['name']] = result
            print(f"--- 任务完成: {task['name']} ---")

        print(f"===== 论文分析完毕: {pdf_path} =====")
        return final_results

    def get_paper_name(self,folder_path: str):
        all_pdf_paths = list(Path(folder_path).glob("*.pdf"))
        return all_pdf_paths



# --- 模块3: 主流程编排器 ---
def main():
    """
    主执行函数，负责编排整个提取流程。
    """
    # --- 配置区 ---
    MY_API_KEY = ""
    paperanalizer = PaperAnalyzer(api_key=MY_API_KEY)
    FILE_PATH = paperanalizer.get_paper_name("testpapers")


    # ================== 任务配置中心 ==================
    # 未来想提取任何新内容，只需在这里添加一个新的任务字典即可！
    TASKS = [
        {
            "name": "摘要提取",
            "target_section": "abstract",  # 告诉程序去哪个章节寻找
            "instruction": "你是一位严谨的科研论文分析师。请仔细阅读以下文本，并从中提取出摘要（Abstract）部分。请只返回完整的摘要内容，不要添加任何额外的解释或标题。如果找不到，请返回'NO_ABSTRACT_FOUND'。",
            "not_found_indicator": "NO_ABSTRACT_FOUND"
        },
        {
            "name": "结论总结",
            "target_section": "conclusion",
            "instruction": "你是一位顶尖的行业研究员。请仔细阅读以下文本，并对其中的结论（Conclusion/Discussion）部分进行总结，提炼出1-3个核心观点。请只返回核心观点，条理清晰。如果找不到，请返回'NO_CONCLUSION_FOUND'。",
            "not_found_indicator": "NO_CONCLUSION_FOUND"
        },
        {
            "name": "方法识别",
            "target_section": "method",
            "instruction": "你是一位AI工程师。请分析以下文本，识别并列出其中提到的所有关键技术、模型或算法名称。请使用列表格式返回。如果找不到，请返回'NO_METHOD_FOUND'。",
            "not_found_indicator": "NO_METHOD_FOUND"
        }
    ]
    # =================================================

    if not MY_API_KEY:
        print("[严重错误] 请在代码中设置您的API密钥 (MY_API_KEY)。")
        return

    # --- 流程开始 ---
    for pdf_file_path in FILE_PATH:
        full_text = paperanalizer._extract_text_from_pdf(file_path=pdf_file_path)
        if not full_text: return

        # 1. 本地预处理，一次性提取所有章节
        all_sections_data = full_text
        print("-" * 40)

        # 2. 遍历任务列表，逐个执行
        final_results = {}
        for task in TASKS:
            print(f"\n===== 开始执行任务: {task['name']} =====")
            # 将任务配置和提取出的章节数据交给处理器
            result = paperanalizer._process_single_task(task, all_sections_data)
            final_results[task['name']] = result
            print(f"===== 任务完成: {task['name']} =====\n结果预览: {result[:150]}...")

        # 3. 打印最终所有结果
        print("\n\n" + "=" * 40)
        print("所有任务执行完毕，最终提取结果汇总如下：")
        print("=" * 40)
        # 使用json库美化输出
        print(json.dumps(final_results, ensure_ascii=False, indent=4))
        with open(f"{pdf_file_path}的相关内容.doc", "w", encoding="utf-8") as f:
            for name, value in final_results.items():
                f.write(f"{name}: {value}\n")







if __name__ == "__main__":
    main()






