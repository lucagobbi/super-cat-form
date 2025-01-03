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


def super_cat_form(Form: type) -> type:
    Form._autopilot = True
    if Form.name is None:
        Form.name = Form.__name__

    if Form.triggers_map is None:
        Form.triggers_map = {
            "start_example": Form.start_examples,
            "description": [f"{Form.name}: {Form.description}"],
        }

    return Form


class PizzaOrder(BaseModel):
    pizza_type: str
    address: str


@super_cat_form
class PizzaForm(SuperCatForm):
    description = "Pizza Order"
    model_class = PizzaOrder
    start_examples = [
        "order a pizza!",
        "I want pizza"
    ]
    stop_examples = [
        "stop pizza order",
        "not hungry anymore",
    ]
    ask_confirm = False

    @form_tool(return_direct=True)
    def get_menu(self):
        """Useful to get the menu. User may ask: what is the menu? Input is always None."""
        return ["Margherita", "Pepperoni", "Hawaiian"]

    @form_tool(return_direct=True)
    def ask_for_daily_promotions(self):
        """Useful to get any daily promotions. User may ask: what are the daily promotions? Input is always None."""
        if datetime.now().weekday() == 0:
            return "Free delivery"
        elif datetime.now().weekday() == 4:
            return "Free Pepperoni"

    def submit(self, form_data):
        return {
            "output": f"Form submitted"
        }
