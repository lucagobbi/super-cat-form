# SuperCatForm

<img src="./assets/supercatform.png" width=400>

[![awesome plugin](https://custom-icon-badges.demolab.com/static/v1?label=&message=awesome+plugin&color=383938&style=for-the-badge&logo=cheshire_cat_ai)](https://)  

SuperCatForm is a powerful, strongly opinionated, and flexible Cheshire Cat plugin that supercharges your conversational forms. Built as an enhancement to the original CatForm, it introduces advanced tools and capabilities for creating more dynamic and responsive user interactions.

## Features

- **Tool calling during form execution**: SuperCatForm enables real-time interactions by allowing tools to be called during form execution. For instance, a restaurant order form can fetch the latest menu, a travel booking form can check live flight availability, or a shopping assistant form can retrieve daily discounts. This turns forms from simple data collectors into smart conversational agents.

- **Nested fields support**: Effortlessly manage complex data structures for richer user interactions with nested fields and data structures.

- **Full Pydantic validation support**: Ensure that validation rules are applied both in the extraction phase (i.e. inserted in the extraction prompt) and during validation phase.

- **Form Events**: Hook into various stages of the form lifecycle to execute custom logic. For example, you can trigger actions when form extraction is completed, or when the form is submitted.

- **JSON schema support** (Coming soon...): Streamline form validation and consistency with schema-based definitions.


## Usage

1. Install the SuperCatForm in your Cheshire Cat instance from the registry.
2. Create a new plugin.
3. Create a form class as you would do with traditional `CatForm`.
4. Define your model class leveraging all the power of Pydantic.
5. Replace the `@form` decorator with `@super_cat_form`.
6. Hook into form events using the `events` attribute.
7. Add useful methods to the class and mark them with `@form_tool`.
8. Have fun!


```python
from typing import Literal, List
from pydantic import BaseModel, Field
from datetime import datetime

from cat.plugins.super_cat_form.super_cat_form import SuperCatForm, form_tool, super_cat_form
from cat.plugins.super_cat_form.super_cat_form_events import FormEvent, FormEventContext


class Address(BaseModel):
    street: str
    city: str


class Pizza(BaseModel):
    pizza_type: str = Field(description="Type of pizza")
    size: Literal["standard", "large"] = Field(default="standard")
    extra_toppings: List[str] = Field(default_factory=list)


class PizzaOrder(BaseModel):
    pizzas: List[Pizza]
    address: Address
    due_date: datetime = Field(description="Datetime when the pizza should be delivered - format YYYY-MM-DD HH:MM")



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
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.events.on(
            FormEvent.EXTRACTION_COMPLETED,
            self.hawaiian_is_not_a_real_pizza
        )
    
    def hawaiian_is_not_a_real_pizza(self, context: FormEventContext):
        ordered_pizzas = context.data.get("pizzas", [])
        for pizza in ordered_pizzas:
            if pizza["pizza_type"] == "Hawaiian":
                self.cat.send_ws_message("Dude... really?", msg_type="chat")

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
        
        form_result = self.form_data_validated

        if form_result is None:
            return {
                "output": "Invalid form data"
            }

        return {
            "output": f"Ok! {form_result.pizzas} will be delivered to {form_result.address} on {form_result.due_date.strftime('%A, %B %d, %Y at %H:%M')}"
        }


```

## Advanced Configuration 🔧

### Custom Prompts

You can customize the prompts used for extraction and tool selection by overriding the default values in your
`SuperCatForm` class. Like this:

```python

@super_cat_form
class MedicalDiagnosticForm(SuperCatForm):
    # Custom NER prompt with domain-specific instructions
    ner_prompt = """You are a medical assistant extracting patient data.
    
    Current form state: {form_data}
    
    Extract following entities:
    - Symptoms
    - Medical history
    - Allergies
    
    {format_instructions}
    """
    
    # Custom tool selection prompt
    tool_prompt = """You are a medical diagnostic assistant. Available tools:
    
    {tools}
    
    Use tools when you need additional information.
    
    {examples}
    """

```

<details>
    <summary>
        Default NER Prompt
    </summary>


```
You are an advanced AI assistant specializing in information extraction and structured data formatting. 
Your task is to extract relevant information from a given text and format it according to a specified JSON structure. 
This extraction is part of a conversational form-filling process.

Here are the key components for this task:

1. Chat History:
<chat_history>
{chat_history}
</chat_history>

2.Form Description:
<form_description>
{form_description}
</form_description>

3. Format Instructions (JSON schema):
<format_instructions>
{format_instructions}
</format_instructions>

Remember:
- The extraction is part of an ongoing conversation, so consider any contextual information that might be relevant.
- Only include information that is explicitly stated or can be directly inferred from the input text.
- If a required field in the JSON schema cannot be filled based on the available information, use null or an appropriate default value as specified in the format instructions.
- Ensure that the output JSON is valid and matches the structure defined in the format instructions.
```
</details>


<details>
    <summary>
        Default Tool Prompt
    </summary>


```
Create a JSON with the correct "action" and "action_input" for form compilation assistance.

Current form data: {form_data}

Available actions: {tools}

CORE RULES:
1. Use specific tools ONLY when explicitly requested by user
2. Default to "no_action" for:
   - Any form filling or ordering intention
   - Direct responses to form questions
   - When no action needed

{examples}

Response Format:
{{
    "action": str,  // One of [{tool_names}, "no_action"]
    "action_input": str | null  // Per action description
}}
```
</details>

### Tool Examples

It is often recommended to provide examples for the usage of tools in the tool prompt.

You can either override the `default_examples` attribute on your `SuperCatForm` class:

```python

@super_cat_form
class HotelBookingForm(SuperCatForm):
    default_examples = """
    Examples:
    "Are pets allowed?" → "are_pets_allowed" (explicit pets request)
    """

    @form_tool
    def are_pets_allowed(self):
        """Useful to check if pets are allowed. User may ask: are pets allowed? Input is always None."""
        return True
```

Or you can use the `examples` parameter on the `@form_tool` decorator, be sure to deactivate the default examples by setting `default_examples = None` on your `SuperCatForm` class:

```python

@super_cat_form
class HotelBookingForm(SuperCatForm):
    default_examples = None

    @form_tool(examples=["Are pets allowed?"])
    def are_pets_allowed(self):
        """Useful to check if pets are allowed. User may ask: are pets allowed? Input is always None."""
        return True
```

<details>
    <summary>
        Default Examples Prompt
    </summary>


```
Examples:
"What's on the menu?" → "get_menu" (explicit menu request)
"I want to order pizza" → "form_compilation" (ordering intention)
"Hi there" → "no_action" (greeting)
```
</details>



## Form Events

You can hook into various stages of the form lifecycle to execute custom logic. For example, you can trigger actions when form extraction is completed, or when the form is submitted.

Events supported:

```python
class FormEvent(Enum):

    # Lifecycle events
    FORM_INITIALIZED = "form_initialized"
    FORM_SUBMITTED = "form_submitted"
    FORM_CLOSED = "form_closed"

    # Extraction events
    EXTRACTION_STARTED = "extraction_started"
    EXTRACTION_COMPLETED = "extraction_completed"

    # Validation events
    VALIDATION_STARTED = "validation_started"
    VALIDATION_COMPLETED = "validation_completed"

    FIELD_UPDATED = "field_updated"

    # Tool events
    TOOL_STARTED = "tool_started"
    TOOL_EXECUTED = "tool_executed"
    TOOL_FAILED = "tool_failed"
```

A form event handler is a regular python function that takes a `FormEventContext` as input:

```python

class FormEventContext(BaseModel):
    timestamp: datetime       # Event occurrence time
    form_id: str              # Form identifier
    event: FormEvent          # Event type
    data: Dict[str, Any]      # Event-specific payload

```