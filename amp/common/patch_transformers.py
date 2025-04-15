import ast
import inspect
from typing import Callable

import astor
from loguru import logger
from transformers.dynamic_module_utils import get_class_in_module


def patch_get_class_in_module(func: Callable) -> Callable:
    """
    动态修改 transformers.dynamic_module_utils.get_class_in_module 函数，
    在返回语句之前插入 func(module, class_name)
    """

    # 获取原始函数的源代码
    ori_get_class_in_module = get_class_in_module
    try:
        original_source = inspect.getsource(ori_get_class_in_module)
    except OSError as e:
        raise RuntimeError("无法获取 'get_class_in_module' 的源代码。") from e

    tree = ast.parse(original_source)

    class InsertFuncCallTransformer(ast.NodeTransformer):
        """
        AST 节点转换器，用于在每个 return 语句之前插入 func(module, class_name) 调用。
        """

        def visit_Return(self, node):
            """
            重写 visit_Return 方法，在 return 语句之前插入 func(module, class_name) 调用。
            """
            # 创建 func(module, class_name) 调用节点
            func_call = ast.Expr(
                value=ast.Call(
                    func=ast.Name(id="func", ctx=ast.Load()),
                    args=[
                        ast.Name(id="module", ctx=ast.Load()),
                    ],
                    keywords=[],
                )
            )

            # 在 return 语句之前插入 func_call
            return [func_call, node]

    # 初始化并应用转换器
    transformer = InsertFuncCallTransformer()
    modified_tree = transformer.visit(tree)
    ast.fix_missing_locations(modified_tree)

    # 将修改后的 AST 转换回源代码
    if hasattr(ast, "unparse"):
        # 对于 Python 3.9 及以上版本
        modified_code = ast.unparse(modified_tree)
    else:
        if astor is not None:
            # 使用 astor 库处理 Python 3.8 及以下版本
            modified_code = astor.to_source(modified_tree)
        else:
            raise RuntimeError(
                "无法将 AST 转换回源代码。请确保安装了 'astor' 库，或使用 Python 3.9 及以上版本。"
            )

    # 准备执行环境，将 func 添加到命名空间中
    ori_globals = ori_get_class_in_module.__globals__
    ori_globals["func"] = func

    # Execute the modified code within the original globals
    exec(modified_code, ori_globals)

    if "get_class_in_module" in ori_globals:
        get_class_in_module_patched = ori_globals["get_class_in_module"]
    elif "om_get_class_in_module" in ori_globals:
        # 适配 openmind
        logger.warning("using 'om_get_class_in_module' function.")
        get_class_in_module_patched = ori_globals["om_get_class_in_module"]
    else:
        raise RuntimeError(
            "无法找到 'get_class_in_module' 或 'om_get_class_in_module' 函数。"
        )

    # Retrieve the patched function from globals

    return get_class_in_module_patched
