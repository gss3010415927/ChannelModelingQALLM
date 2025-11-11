from prompts.stuff_chain import StuffChainPrompts
from prompts.refine_chain import RefineChainPrompts


class PromptFactory:
    """提示词工厂类"""

    @staticmethod
    def get_prompt(chain_type: str, prompt_name: str = None):
        if chain_type == "stuff":
            return StuffChainPrompts.get_prompt()
        elif chain_type == "refine":
            if prompt_name == "initial":
                return RefineChainPrompts.get_initial_prompt()
            elif prompt_name == "refine":
                return RefineChainPrompts.get_refine_prompt()
            elif prompt_name == "document_prompt":
                return RefineChainPrompts.get_document_prompt()
            else:
                raise ValueError(
                    f"Unsupported prompt name for refine chain: {prompt_name}")
        else:
            raise ValueError(f"Unsupported chain type: {chain_type}")
