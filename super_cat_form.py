import inspect
from functools import wraps
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

    def __init__(self, cat):
        super().__init__(cat)
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

    def _ner(self) -> Dict:
        """
        Executes NER using LangChain JsonOutputParser on current form

        Returns:
            Dict: NER result
        """
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
        parser = JsonOutputParser(pydantic_object=self.model_class)
        prompt = PromptTemplate(
            template=self.ner_prompt,
            input_variables=list(prompt_params.keys()),
            partial_variables={"format_instructions":
                                   parser.get_format_instructions()},
        )
        chain = prompt | self.cat._llm | parser
        ner_result = chain.invoke(prompt_params)
        self.events.emit(
            FormEvent.EXTRACTION_COMPLETED,
            data=ner_result,
            form_id=self.name
        )
        return ner_result

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
            output_model = self._ner()
        except Exception as e:
            output_model = {}
            log.warning(e)

        return output_model

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
                return self.submit(self._model)
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

        if self.check_exit_intent():
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
                return self.submit(self._model)

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
