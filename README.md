# SuperCatForm

<img src="./assets/supercatform.png" width=400>

[![awesome plugin](https://custom-icon-badges.demolab.com/static/v1?label=&message=awesome+plugin&color=383938&style=for-the-badge&logo=cheshire_cat_ai)](https://)  

SuperCatForm is a powerful, strongly opinionated, and flexible Cheshire Cat plugin that supercharges your conversational forms. Built as an enhancement to the original CatForm, it introduces advanced tools and capabilities for creating more dynamic and responsive user interactions.

## Features

- **Tool calling during form execution**: SuperCatForm enables real-time interactions by allowing tools to be called during form execution. For instance, a restaurant order form can fetch the latest menu, a travel booking form can check live flight availability, or a shopping assistant form can retrieve daily discounts. This turns forms from simple data collectors into smart conversational agents.


- **JSON schema support** (Coming soon...): Streamline form validation and consistency with schema-based definitions.


- **Nested fields support** (Coming soon...): Effortlessly manage complex data structures for richer user interactions.

## Usage

1. Install the SuperCatForm in your Cheshire Cat instance from the registry.
2. Create a new plugin.
3. Create a form class as you would do with traditional `CatForm`.
4. Replace the `@form` decorator with `@super_cat_form`.
5. Add useful methods to the class and mark them with `@form_tool`.
6. Have fun!


```python
from pydantic import BaseModel
from datetime import datetime

from cat.plugins.super_cat_form.super_cat_form import SuperCatForm, form_tool, super_cat_form


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
            "output": f"Form submitted: {form_data}"
        }


```
