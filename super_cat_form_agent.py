from typing import Dict, Callable
import traceback
import inspect

from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import SystemMessagePromptTemplate
from langchain_core.runnables import RunnableConfig, RunnableLambda

from cat.agents import BaseAgent, AgentOutput
from cat.looking_glass.output_parser import ChooseProcedureOutputParser, LLMAction
from cat.looking_glass.callbacks import ModelInteractionHandler
from cat.plugins.super_cat_form import prompts
from cat.log import log
from cat import utils

class SuperCatFormAgent(BaseAgent):
    """Agent that executes form-based tools based on conversation context."""

    def __init__(self, form_tools: Dict[str, Callable], form_instance):
        self.form_tools = form_tools
        self.form_instance = form_instance

    def execute(self, stray) -> AgentOutput:
        if not self.form_tools:
            return AgentOutput()
        try:
            return self._process_tools(stray)
        except Exception as e:
            log.error(f"Error in agent execution: {str(e)}")
            traceback.print_exc()
            return AgentOutput()

    def _process_tools(self, stray) -> AgentOutput:

        llm_action = self._execute_tool_selection_chain(stray, prompts.TOOL_PROMPT)

        log.debug(f"Selected tool: {llm_action}")

        result = self._execute_tool(llm_action)

        return result

    def _execute_tool_selection_chain(self, stray, prompt_template: str) -> LLMAction:
        """Execute the LLM chain to select appropriate tool."""
        prompt_vars = self._prepare_prompt_variables()

        prompt_vars, prompt_template = utils.match_prompt_variables(
            prompt_vars,
            prompt_template
        )

        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(template=prompt_template),
                *(stray.langchainfy_chat_history()),
            ]
        )

        chain = (
                prompt
                | RunnableLambda(lambda x: utils.langchain_log_prompt(x, "TOOL PROMPT"))
                | stray._llm
                | RunnableLambda(lambda x: utils.langchain_log_output(x, "TOOL PROMPT OUTPUT"))
                | ChooseProcedureOutputParser()
        )

        return chain.invoke(
            prompt_vars,
            config=RunnableConfig(
                callbacks=[ModelInteractionHandler(stray, self.__class__.__name__)]
            )
        )

    def _execute_tool(self, llm_action: LLMAction) -> AgentOutput:
        """Execute the selected tool and return results."""

        if not llm_action.action:
            return AgentOutput(output="")

        chosen_procedure = self.form_tools.get(llm_action.action)

        if not chosen_procedure:
            log.error(f"Unknown tool: {llm_action.action}")
            return AgentOutput(output="")

        try:
            bound_method = chosen_procedure.__get__(self.form_instance, self.form_instance.__class__)
            sig = inspect.signature(chosen_procedure)
            params = list(sig.parameters.keys())
            tool_output = (
                bound_method() if len(params) == 1 else bound_method(llm_action.action_input)
            )
            return AgentOutput(
                output=str(tool_output),
                return_direct=chosen_procedure._return_direct,
                intermediate_steps=[
                    ((llm_action.action, llm_action.action_input), tool_output)
                ]
            )
        except Exception as e:
            log.error(
                f"Error executing form tool `{chosen_procedure.__name__}`: {str(e)}"
            )
            traceback.print_exc()

        return AgentOutput(output="")

    def _prepare_prompt_variables(self) -> Dict[str, str]:
        """Prepare variables for the prompt template."""
        return {
            "form_data": str(self.form_instance.form_data),
            "tools": "\n".join(
                f'- "{tool.__name__}": {tool.__doc__.strip()}'
                for tool in self.form_tools.values()
            ),
            "tool_names": '"' + '", "'.join(self.form_tools.keys()) + '"',
            # "examples": self._generate_examples() # TODO add examples
        }
