import inspect
from functools import wraps
from pydantic import BaseModel

from cat.experimental.form import form, CatForm, CatFormState
from cat.log import log


def form_tool(func):
    """Decorator to register methods as form tools."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        return func(self, *args, **kwargs)

    wrapper._is_form_tool = True
    return wrapper




class SuperCatForm(CatForm):

    @classmethod
    def _get_form_tools(cls):
        """
        Get all methods of the class that are decorated with @form_tool.
        Including inherited methods.
        """
        form_tools = []
        for name, func in inspect.getmembers(cls):
            if inspect.isfunction(func) or inspect.ismethod(func):
                if getattr(func, '_is_form_tool', False):
                    form_tools.append(name)
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
            # Execute form tools
            for tool in self._get_form_tools():
                log.critical("Executing form tool: " + tool)
                getattr(self, tool)()

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

    @form_tool
    def get_form_data(self):
        return self.form_data

    def submit(self, form_data):
        return {
            "output": f"Form submitted"
        }