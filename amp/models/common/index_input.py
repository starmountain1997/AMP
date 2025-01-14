import ast
import inspect
import textwrap

from loguru import logger


class IndexPutTransformer(ast.NodeTransformer):
    def visit_Assign(self, node: ast.Assign):
        """
        Look for:
            inputs_embeds = inputs_embeds.index_put([...], images_features)

        Replace with two statements:
            mask = (token_type_ids == VISION_TOKEN_TYPE).int()
            inputs_embeds = (inputs_embeds * (1 - mask)) + images_features * mask
        """
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_name = node.targets[0].id
            if target_name == "inputs_embeds":
                if (
                    isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == "index_put"
                    and isinstance(node.value.func.value, ast.Name)
                    and node.value.func.value.id == "inputs_embeds"
                ):
                    logger.warning("patching index_put to mask_add")
                    mask_assign = ast.parse(
                        "mask = (token_type_ids == VISION_TOKEN_TYPE).int()"
                    ).body[0]

                    embeds_assign = ast.parse(
                        "inputs_embeds = (inputs_embeds * (1 - mask)) + images_features * mask"
                    ).body[0]

                    # Return a list of new statements to replace the single `Assign`
                    return [mask_assign, embeds_assign]
        return node


def patch_index_input(mod, func):
    source_code = inspect.getsource(func)
    source_code = textwrap.dedent(source_code)
    transformer = IndexPutTransformer()
    new_tree = transformer.visit(ast.parse(source_code))
    new_code = ast.unparse(new_tree)
    exec(new_code, mod.__dict__)
