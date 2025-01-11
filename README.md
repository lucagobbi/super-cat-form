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
