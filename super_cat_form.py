import inspect
from functools import wraps
import re
from typing import Dict, Optional, Type, List
from pydantic import BaseModel, ValidationError

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser

from cat.looking_glass.callbacks import NewTokenHandler
from cat.experimental.form import form, CatForm, CatFormState
from cat.plugins.super_cat_form.super_cat_form_agent import SuperCatFormAgent
from cat.plugins.super_cat_form.super_cat_form_events import FormEventManager, FormEvent, FormEventContext
from cat.plugins.super_cat_form import prompts
from cat.log import log
from cat import utils

from cat.looking_glass.callbacks import ModelInteractionHandler


def form_tool(func=None, *, return_direct=False, examples=None):

    if examples is None:
        examples = []

    if func is None:
        return lambda f: form_tool(f, return_direct=return_direct, examples=examples)

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        return func(self, *args, **kwargs)

    wrapper._is_form_tool = True
    wrapper._return_direct = return_direct
    wrapper._examples = examples
    return wrapper


class SuperCatForm(CatForm):
    """
    SuperCatForm is the CatForm class that extends the functionality of the original CatForm class.
    """
    ner_prompt = prompts.DEFAULT_NER_PROMPT
    tool_prompt = prompts.DEFAULT_TOOL_PROMPT
    default_examples = prompts.DEFAULT_TOOL_EXAMPLES

    inside_forms = []

    def __init__(self, cat, previous_form=None):
        super().__init__(cat)

        # This iniziale the forms in self.inside_forms,
        self.initialize_inside_forms()

        self.tool_agent = SuperCatFormAgent(self)
        self.events = FormEventManager()
        self._setup_default_handlers()
        # This hack to ensure backward compatibility with version pre-1.8.0
        self._legacy_version = 'model' in inspect.signature(super().validate).parameters
        self.events.emit(
            FormEvent.FORM_INITIALIZED,
            data={},
            form_id=self.name
        )
        self.cat.llm = self.super_llm
        self.previous_form = previous_form

    def __reset_active_form(self, *args, **kwargs):
        """
        Reset the active form to the previous form, if exists.
        """
        if self.previous_form is not None:
            self.cat.working_memory.active_form = self.previous_form

    def super_llm(self, prompt: str | ChatPromptTemplate, params: dict = None, stream: bool = False) -> str:

        callbacks = []
        if stream:
            callbacks.append(NewTokenHandler(self.cat))

        caller = utils.get_caller_info()
        callbacks.append(ModelInteractionHandler(self.cat, caller or "StrayCat"))

        if isinstance(prompt, str):
            prompt = ChatPromptTemplate(
                messages=[
                    # Use HumanMessage instead of SystemMessage for wide-range compatibility
                    HumanMessage(content=prompt)
                ]
            )

        chain = (
                prompt
                | RunnableLambda(lambda x: utils.langchain_log_prompt(x, f"{caller} prompt"))
                | self.cat._llm
                | RunnableLambda(lambda x: utils.langchain_log_output(x, f"{caller} prompt output"))
                | StrOutputParser()
        )

        output = chain.invoke(
            params or {},
            config=RunnableConfig(callbacks=callbacks)
        )

        return output

    def _setup_default_handlers(self):
        """Setup default event handlers for logging"""
        for event in FormEvent:
            self.events.on(event, self._log_event)

        # Setup event handler for inside form creation
        self.events.on(
            FormEvent.INSIDE_FORM_ACTIVE,
            self._on_inside_create_form
        )

        # Setup event handler for form inside closure
        self.events.on(
            FormEvent.INSIDE_FORM_CLOSED,
            self._on_inside_form_closed
        )

        # Setup event handler for form closure
        self.events.on(
            FormEvent.FORM_CLOSED,
            self._on_form_closed
        )

        # Reset the active form when the form is submitted or closed
        self.events.on(
            FormEvent.FORM_SUBMITTED,
            self.__reset_active_form
        )
        self.events.on(
            FormEvent.FORM_CLOSED,
            self.__reset_active_form
        )

    def _log_event(self, event: FormEventContext):
        log.debug(f"Form {self.name}: {event.event.name} - {event.data}")

    def _get_validated_form_data(self) -> Optional[BaseModel]:
        """
        Safely attempts to get validated form data.
        Returns None if the form is incomplete or invalid.

        Returns:
            Optional[BaseModel]: Validated Pydantic model if successful, None otherwise
        """
        try:
            return self.model_getter()(**self._model)
        except ValidationError:
            return None

    @classmethod
    def get_form_tools(cls):
        """
        Get all methods of the class that are decorated with @form_tool.
        """
        form_tools = {}
        for name, func in inspect.getmembers(cls):
            if inspect.isfunction(func) or inspect.ismethod(func):
                if getattr(func, '_is_form_tool', False):
                    form_tools[name] = func
        return form_tools

    @staticmethod
    def format_class_name(name):
        """
        Formats a class name into snake_case by inserting underscores before capital letters
        and converting the result to lowercase.

        Args:
            name (str): The class name to format.

        Returns:
            str: The formatted name in snake_case.
        """
        def replacement(match):
            """
            Helper function to determine the replacement for regex matches.

            Args:
                match (re.Match): The regex match object.

            Returns:
                str: The replacement string.
            """
            # If the match is from the second pattern (?<!^)(?=[A-Z]), insert an underscore
            if match.group(0) == '':
                return '_'
            # If the match is from the first pattern ([A-Z]+)([A-Z][a-z]), insert an underscore between groups
            return match.group(1) + '_' + match.group(2)

        # Regex pattern to handle
        pattern = r'([A-Z]+)([A-Z][a-z])|(?<!^)(?=[A-Z])'

        # Apply the regex substitution, convert to lowercase, and replace spaces with underscores
        return re.sub(pattern, replacement, name).lower().replace(" ", "_")

    @classmethod
    def initialize_inside_forms(cls):
        """
        Initializes inside forms for the current form. For each form class in `cls.inside_forms`,
        if it is a subclass of `CatForm`, create a dynamic method decorated with `form_tool`
        that starts the inside form when called.
        """
        for form_class in cls.inside_forms:
            if issubclass(form_class, CatForm):
                
                # Format the form class name into snake_case
                formatted_form_name = cls.format_class_name(form_class.name or form_class.__name__)
                tool_name = f"start_form_{formatted_form_name}"

                # Define a dynamic method to start the inside form
                def tool_start_inside_form(self, *args):
                    return self.start_inside_form(form_class)

                # Set the docstring of the dynamic method to include the example
                # All examples are joined with " or " in the tool docstring
                tool_start_inside_form.__doc__ = " or ".join(form_class.start_examples) + ". Input is always None."

                # Set the name of the dynamic form method
                tool_start_inside_form.__name__ = tool_name

                # Wrap the dynamic method as a form tool
                wrapped_form_tool = form_tool(
                    func=tool_start_inside_form,
                    return_direct=True
                )

                # Set the methods to the class
                setattr(cls, tool_name, wrapped_form_tool)

    def update(self):
        """
        Version-compatible update method that works with both old and new CatForm versions.
        Ensures _model is always a dictionary.
        """

        old_model = self._model.copy() if self._model is not None else {}

        # Extract and sanitize new data
        json_details = self.extract()
        json_details = self.sanitize(json_details)
        merged_model = old_model | json_details

        if self._legacy_version:
            # old version: validate returns the updated model
            validated_model = self.validate(merged_model)
            # ensure we never set None as the model
            self._model = validated_model if validated_model is not None else {}
        else:
            # new version: set model first, then validate
            self._model = merged_model
            self.validate()

        # ensure self._model is never None
        if self._model is None:
            self._model = {}

        # emit events for updated fields
        updated_fields = {
            k: v for k, v in self._model.items()
            if k not in old_model or old_model[k] != v
        }

        if updated_fields:
            self.events.emit(
                FormEvent.FIELD_UPDATED,
                {
                    "fields": updated_fields,
                    "old_values": {k: old_model.get(k) for k in updated_fields}
                },
                self.name
            )

    def sanitize(self, model: Dict) -> Dict:
        """
        Sanitize the model while preserving nested structures.
        Only removes explicitly null values.

        Args:
            model: Dictionary containing form data

        Returns:
            Dict: Sanitized form data
        """

        if "$defs" in model:
            del model["$defs"]

        def _sanitize_nested(data):
            if isinstance(data, dict):
                return {
                    k: _sanitize_nested(v)
                    for k, v in data.items()
                    if v not in ("None", "null", "lower-case", "unknown", "missing")
                }
            return data

        return _sanitize_nested(model)

    def validate(self, model=None):
        """
        Override the validate method to properly handle nested structures
        while preserving partial data.
        """
        self.events.emit(
            FormEvent.VALIDATION_STARTED,
            {"model": self._model},
            self.name
        )

        self._missing_fields = []
        self._errors = []

        try:
            if self._legacy_version and model is not None:
                validated_model = self.model_getter()(**model).model_dump(mode="json")
                self._state = CatFormState.COMPLETE
                return validated_model
            else:
                # New version: validate self._model
                self.model_getter()(**self._model)
                self._state = CatFormState.COMPLETE


        except ValidationError as e:
            for error in e.errors():
                field_path = '.'.join(str(loc) for loc in error['loc'])
                if error['type'] == 'missing':
                    self._missing_fields.append(field_path)
                else:
                    self._errors.append(f'{field_path}: {error["msg"]}')

            self._state = CatFormState.INCOMPLETE

            if self._legacy_version and model is not None:
                return model
        finally:
            self.events.emit(
                FormEvent.VALIDATION_COMPLETED,
                {
                    "model": self._model,
                    "missing_fields": self._missing_fields,
                    "errors": self._errors
                },
                self.name
            )

    def extract(self):
        """
        Override the extract method to include NER with LangChain JsonOutputParser
        """
        try:
            self.events.emit(
                FormEvent.EXTRACTION_STARTED,
                data={
                    "chat_history": self.cat.stringify_chat_history(),
                    "form_data": self.form_data
                },
                form_id=self.name
            )
            prompt_params = {
                "chat_history": self.cat.stringify_chat_history(),
                "form_description": f"{self.name} - {self.description}"
            }
            parser = JsonOutputParser(pydantic_object=self.model_getter())
            prompt = PromptTemplate(
                template=self.ner_prompt,
                input_variables=list(prompt_params.keys()),
                partial_variables={"format_instructions":
                                       parser.get_format_instructions()},
            )
            chain = prompt | self.cat._llm | parser
            output_model = chain.invoke(prompt_params)
            self.events.emit(
                FormEvent.EXTRACTION_COMPLETED,
                data=output_model,
                form_id=self.name
            )
        except Exception as e:
            output_model = {}
            log.error(e)

        return output_model

    def submit_close(self, form_data):
        """
        Submit the actual form and reset the previous, if exists
        """

        if self.previous_form is not None:
            self.__reset_active_form()

            self.previous_form.events.emit(
                FormEvent.INSIDE_FORM_CLOSED,
                {
                    "form_data": form_data,
                    "output": self.submit(form_data)
                },
                self.name
            )

            # Return message of the external (old) form
            return self.previous_form.message()

        # By default, return the submit output
        return self.submit(form_data)

    def start_inside_form(self, form_class):
        """
        Create and start a new form instance inside the current form.
        """

        new_form_instance = form_class(
            cat=self.cat,
            previous_form=self
        )

        # Set as active form
        self.cat.working_memory.active_form = new_form_instance

        self.events.emit(
            FormEvent.INSIDE_FORM_ACTIVE,
            {
                "instance": new_form_instance
            },
            self.name
        )

        # Return the first message of the new form
        return new_form_instance.next()["output"]

    # Event handlers
    def _on_inside_create_form(self, context: FormEventContext):
        """
        Called when a new inside form is created.
        """
        log.debug(f"[EVENT: _on_inside_create_form] inside form in {self.name} created")

        form_class = context.data.get("instance")
        log.debug(f"Creating inside form: {form_class}")

    def _on_inside_form_closed(self, context: FormEventContext):
        """
        Called when the form is closed.
        """

        submit_output = context.data.get("output")
        form_data = context.data.get("form_data")

        # Send the submit output to chat
        self.cat.send_chat_message(submit_output["output"])

    def _on_form_closed(self, form_data):
        """
        Called when the form is closed.
        """
        log.debug(f"[EVENT: _on_form_closed] form {self.name} closed")

        if self.previous_form is not None:

            self.previous_form.events.emit(
                FormEvent.INSIDE_FORM_CLOSED,
                {
                    "form_data": form_data,
                    "output": self.message_closed(force=True)
                },
                self.name
            )

    def next(self):

        if self._state == CatFormState.WAIT_CONFIRM:
            if self.confirm():
                self._state = CatFormState.CLOSED
                self.events.emit(
                    FormEvent.FORM_SUBMITTED,
                    {
                        "form_data": self.form_data
                    },
                    self.name
                )
                return self.submit_close(self._model)
            else:
                if self.check_exit_intent():
                    self._state = CatFormState.CLOSED
                    self.events.emit(
                        FormEvent.FORM_CLOSED,
                        {
                            "form_data": self.form_data
                        },
                        self.name
                    )
                else:
                    self._state = CatFormState.INCOMPLETE

        if self.check_exit_intent() and not self._state == CatFormState.CLOSED:
            self._state = CatFormState.CLOSED
            self.events.emit(
                FormEvent.FORM_CLOSED,
                {
                    "form_data": self.form_data
                },
                self.name
            )

        if self._state == CatFormState.INCOMPLETE:

            # Execute agent if form tools are present
            if len(self.get_form_tools()) > 0:
                agent_output = self.tool_agent.execute(self.cat)
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
                return self.submit_close(self._model)

        return self.message()

    def model_getter(self) -> Type[BaseModel]:
        """
        Override for backward compatibility with older CatForm versions where model_getter
        might not be implemented. This method simply returns model_class, which maintains
        identical functionality while ensuring the method exists in legacy scenarios (pre 1.8.0).
        """
        return self.model_class

    @property
    def form_data(self) -> Dict:
        return self._model

    @property
    def form_data_validated(self) -> Optional[BaseModel]:
        return self._get_validated_form_data()


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
