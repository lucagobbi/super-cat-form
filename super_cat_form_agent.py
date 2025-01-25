from typing import Dict
import traceback
import inspect

from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import SystemMessagePromptTemplate
from langchain_core.runnables import RunnableConfig, RunnableLambda

from cat.plugins.super_cat_form.super_cat_form_events import FormEvent
from cat.agents import BaseAgent, AgentOutput
from cat.looking_glass.output_parser import ChooseProcedureOutputParser, LLMAction
from cat.looking_glass.callbacks import ModelInteractionHandler
from cat.log import log
from cat import utils

class SuperCatFormAgent(BaseAgent):
    """Agent that executes form-based tools based on conversation context."""

    def __init__(self, form_instance):
        self.form_tools = form_instance.get_form_tools()
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

        llm_action = self._execute_tool_selection_chain(stray, self.form_instance.tool_prompt)

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

        if not llm_action.action or llm_action.action == "no_action":
            return AgentOutput(output="")

        chosen_procedure = self.form_tools.get(llm_action.action)

        if not chosen_procedure:
            log.error(f"Unknown tool: {llm_action.action}")
            return AgentOutput(output="")

        try:
            self.form_instance.events.emit(
                event=FormEvent.TOOL_STARTED,
                data={
                    "tool": llm_action.action,
                    "tool_input": llm_action.action_input
                },
                form_id=self.form_instance.name
            )
            bound_method = chosen_procedure.__get__(self.form_instance, self.form_instance.__class__)
            sig = inspect.signature(chosen_procedure)
            params = list(sig.parameters.keys())
            tool_output = (
                bound_method() if len(params) == 1 else bound_method(llm_action.action_input)
            )
            self.form_instance.events.emit(
                event=FormEvent.TOOL_EXECUTED,
                data={
                    "tool": llm_action.action,
                    "tool_input": llm_action.action_input,
                    "tool_output": tool_output
                },
                form_id=self.form_instance.name
            )
            return AgentOutput(
                output=str(tool_output),
                return_direct=chosen_procedure._return_direct,
                intermediate_steps=[
                    ((llm_action.action, llm_action.action_input), tool_output)
                ]
            )
        except Exception as e:
            self.form_instance.events.emit(
                event=FormEvent.TOOL_FAILED,
                data={
                    "tool": llm_action.action,
                    "tool_input": llm_action.action_input,
                    "error": str(e)
                },
                form_id=self.form_instance.name
            )
            log.error(
                f"Error executing form tool `{chosen_procedure.__name__}`: {str(e)}"
            )
            traceback.print_exc()

        return AgentOutput(output="")

    def _generate_examples(self) -> str:
        default_examples = self.form_instance.default_examples

        if default_examples:
            return default_examples

        default_examples = "Examples:\n" + "\n".join(
            f"{k}: {v._examples}"
            for k, v in self.form_tools.items()
        )

        return default_examples


    def _prepare_prompt_variables(self) -> Dict[str, str]:
        """Prepare variables for the prompt template."""
        return {
            "form_data": str(self.form_instance.form_data),
            "tools": "\n".join(
                f'- "{tool.__name__}": {tool.__doc__.strip()}'
                for tool in self.form_tools.values()
            ),
            "tool_names": '"' + '", "'.join(self.form_tools.keys()) + '"',
            "examples": self._generate_examples()
        }
