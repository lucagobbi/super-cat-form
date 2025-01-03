import inspect
from functools import wraps
from pydantic import BaseModel

from cat.experimental.form import form, CatForm, CatFormState
from cat.plugins.super_cat_form.super_cat_form_agent import SuperCatFormAgent
from cat.log import log

from datetime import datetime


def form_tool(func=None, *, return_direct=False):
    if func is None:
        return lambda f: form_tool(f, return_direct=return_direct)

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        return func(self, *args, **kwargs)

    wrapper._is_form_tool = True
    wrapper._return_direct = return_direct
    return wrapper


class SuperCatForm(CatForm):
    """
    SuperCatForm is the CatForm class that extends the functionality of the original CatForm class.
    """

    def __init__(self, cat):
        super().__init__(cat)
        self.tool_agent = SuperCatFormAgent(self._get_form_tools(), self)

    @classmethod
    def _get_form_tools(cls):
        """
        Get all methods of the class that are decorated with @form_tool.
        """
        form_tools = {}
        for name, func in inspect.getmembers(cls):
            if inspect.isfunction(func) or inspect.ismethod(func):
                if getattr(func, '_is_form_tool', False):
                    form_tools[name] = func
        return form_tools

    def next(self):

        if self._state == CatFormState.WAIT_CONFIRM:
            if self.confirm():
                self._state = CatFormState.CLOSED
                return self.submit(self._model)
            else:
                if self.check_exit_intent():
                    self._state = CatFormState.CLOSED
                else:
                    self._state = CatFormState.INCOMPLETE

        if self.check_exit_intent():
            self._state = CatFormState.CLOSED

        if self._state == CatFormState.INCOMPLETE:

            # Execute agent if form tools are present
            if len(self._get_form_tools()) > 0:
                agent_output = self.tool_agent.execute(self.cat)
                log.critical(f"Agent output: {agent_output}")
                if agent_output.output:
                    if agent_output.return_direct:
                        return {"output": agent_output.output}
                self.update()
            else:
                self.update()

        if self._state == CatFormState.COMPLETE:
            if self.ask_confirm:
                self._state = CatFormState.WAIT_CONFIRM
            else:
                self._state = CatFormState.CLOSED
                return self.submit(self._model)

        return self.message()

    @property
    def form_data(self):
        return self._model


def super_cat_form(form: SuperCatForm) -> SuperCatForm:
    """
    Decorator to mark a class as a SuperCatForm.
    """
    form._autopilot = True
    if form.name is None:
        form.name = form.__name__

    if form.triggers_map is None:
        form.triggers_map = {
            "start_example": form.start_examples,
            "description": [f"{form.name}: {form.description}"],
        }

    return form
