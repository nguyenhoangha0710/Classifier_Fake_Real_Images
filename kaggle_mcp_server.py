import os
import sys
import json
import logging
from typing import Optional

# Reconfigure stdout/stderr to utf-8 for Windows console safety
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from mcp.server.mcpserver import MCPServer
import kaggle

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("kaggle-mcp-server")

mcp = MCPServer("kaggle-mcp-server")

def get_api() -> kaggle.KaggleApi:
    api = kaggle.KaggleApi()
    api.authenticate()
    return api

@mcp.tool()
def auth_status() -> str:
    """Kiểm tra trạng thái xác thực API Kaggle."""
    try:
        api = get_api()
        user = api.get_config_value("username") or "hoangha071005"
        return f"SUCCESS: Đã xác thực thành công với tài khoản Kaggle: {user}"
    except Exception as e:
        return f"ERROR: Không thể xác thực Kaggle API: {str(e)}"

@mcp.tool()
def search_datasets(query: str, max_results: int = 5) -> str:
    """Tìm kiếm Datasets trên Kaggle theo từ khóa."""
    try:
        api = get_api()
        datasets = api.dataset_list(search=query)
        results = []
        for d in datasets[:max_results]:
            results.append({
                "ref": getattr(d, 'ref', str(d)),
                "title": getattr(d, 'title', ''),
                "size": getattr(d, 'size', ''),
                "download_count": getattr(d, 'downloadCount', 0)
            })
        return json.dumps(results, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"ERROR khi tìm dataset: {str(e)}"

@mcp.tool()
def download_dataset(dataset_handle: str, target_dir: str = "./dataset") -> str:
    """Tải và giải nén Dataset từ Kaggle về thư mục local."""
    try:
        api = get_api()
        os.makedirs(target_dir, exist_ok=True)
        api.dataset_download_files(dataset_handle, path=target_dir, unzip=True)
        return f"SUCCESS: Đã tải dataset '{dataset_handle}' vào '{target_dir}' thành công!"
    except Exception as e:
        return f"ERROR khi tải dataset: {str(e)}"

@mcp.tool()
def search_competitions(query: str) -> str:
    """Tìm kiếm cuộc thi (Competitions) trên Kaggle."""
    try:
        api = get_api()
        comps = api.competitions_list(search=query)
        results = []
        for c in comps:
            results.append({
                "ref": getattr(c, 'ref', str(c)),
                "title": getattr(c, 'title', ''),
                "category": getattr(c, 'category', ''),
                "reward": getattr(c, 'reward', '')
            })
        return json.dumps(results, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"ERROR khi tìm competition: {str(e)}"

@mcp.tool()
def download_competition_files(competition_name: str, target_dir: str = "./dataset") -> str:
    """Tải file dữ liệu cuộc thi từ Kaggle."""
    try:
        api = get_api()
        os.makedirs(target_dir, exist_ok=True)
        api.competition_download_files(competition_name, path=target_dir)
        return f"SUCCESS: Đã tải dữ liệu cuộc thi '{competition_name}' vào '{target_dir}'!"
    except Exception as e:
        return f"ERROR khi tải dữ liệu cuộc thi: {str(e)}"

@mcp.tool()
def check_kernel_status(kernel_slug: str) -> str:
    """Kiểm tra trạng thái chạy Notebook/Kernel trên Kaggle (ví dụ: 'username/notebook-name')."""
    try:
        api = get_api()
        status_res = api.kernels_status(kernel_slug)
        return json.dumps({
            "kernel": kernel_slug,
            "status": getattr(status_res, 'status', str(status_res)),
            "failureMessage": getattr(status_res, 'failureMessage', None)
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"ERROR khi kiểm tra trạng thái kernel: {str(e)}"

@mcp.tool()
def download_kernel_output(kernel_slug: str, target_dir: str = "./outputs") -> str:
    """Tải kết quả / file output (weights, submission.csv) sau khi Notebook Kaggle chạy xong."""
    try:
        api = get_api()
        os.makedirs(target_dir, exist_ok=True)
        api.kernels_output(kernel_slug, path=target_dir)
        return f"SUCCESS: Đã tải output từ kernel '{kernel_slug}' vào '{target_dir}'!"
    except Exception as e:
        return f"ERROR khi tải output kernel: {str(e)}"

@mcp.tool()
def submit_competition(competition_name: str, file_path: str, message: str = "Submission via MCP") -> str:
    """Nộp file kết quả (submission file) cho một cuộc thi trên Kaggle."""
    try:
        api = get_api()
        api.competition_submit(file_path=file_path, message=message, competition=competition_name)
        return f"SUCCESS: Đã nộp file '{file_path}' cho cuộc thi '{competition_name}'!"
    except Exception as e:
        return f"ERROR khi nộp bài: {str(e)}"

if __name__ == "__main__":
    mcp.run()
