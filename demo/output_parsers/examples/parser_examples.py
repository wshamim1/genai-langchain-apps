"""
Output Parser Examples
Demonstrates using different output parsers for structured data extraction.
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List

# Load environment variables
env_path = Path(__file__).parent.parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from langchain_openai import ChatOpenAI
from langchain.output_parsers import ResponseSchema
from src.output_parsers import PydanticOutputParser, JSONOutputParser, StructuredOutputParser


# Define Pydantic models for examples
class Person(BaseModel):
    """Information about a person."""
    name: str = Field(description="The person's full name")
    age: int = Field(description="The person's age in years")
    occupation: str = Field(description="The person's occupation or job title")
    hobbies: List[str] = Field(description="List of the person's hobbies")


class Product(BaseModel):
    """Information about a product."""
    name: str = Field(description="Product name")
    price: float = Field(description="Product price in USD")
    category: str = Field(description="Product category")
    in_stock: bool = Field(description="Whether the product is in stock")
    rating: float = Field(description="Product rating from 0 to 5")


class MovieReview(BaseModel):
    """A movie review with sentiment analysis."""
    movie_title: str = Field(description="Title of the movie")
    rating: int = Field(description="Rating from 1 to 10")
    sentiment: str = Field(description="Overall sentiment: positive, negative, or neutral")
    summary: str = Field(description="Brief summary of the review")
    would_recommend: bool = Field(description="Whether the reviewer would recommend the movie")


def example_pydantic_parser_person():
    """Example using Pydantic parser to extract person information."""
    print("\n=== Pydantic Parser - Person Information ===")
    
    # Initialize parser
    parser = PydanticOutputParser(pydantic_object=Person)
    
    # Get format instructions
    format_instructions = parser.get_format_instructions()
    print(f"Format Instructions:\n{format_instructions[:200]}...\n")
    
    # Create prompt
    prompt = f"""Extract information about the following person:
    
John Smith is a 35-year-old software engineer who loves hiking, photography, and playing guitar.

{format_instructions}
"""
    
    # Get response from LLM
    llm = ChatOpenAI(temperature=0)
    response = llm.invoke(prompt)
    
    print(f"LLM Response:\n{response.content}\n")
    
    # Parse the response
    person = parser.parse(response.content)
    
    print(f"Parsed Person:")
    print(f"  Name: {person.name}")
    print(f"  Age: {person.age}")
    print(f"  Occupation: {person.occupation}")
    print(f"  Hobbies: {', '.join(person.hobbies)}")
    
    return person


def example_pydantic_parser_product():
    """Example using Pydantic parser to extract product information."""
    print("\n=== Pydantic Parser - Product Information ===")
    
    parser = PydanticOutputParser(pydantic_object=Product)
    
    prompt = f"""Extract product information from this description:

The UltraBook Pro is a high-performance laptop priced at $1,299.99. 
It's in the Electronics category and currently in stock. 
Customers have rated it 4.5 out of 5 stars.

{parser.get_format_instructions()}
"""
    
    llm = ChatOpenAI(temperature=0)
    response = llm.invoke(prompt)
    
    product = parser.parse(response.content)
    
    print(f"Parsed Product:")
    print(f"  Name: {product.name}")
    print(f"  Price: ${product.price}")
    print(f"  Category: {product.category}")
    print(f"  In Stock: {product.in_stock}")
    print(f"  Rating: {product.rating}/5.0")
    
    return product


def example_json_parser():
    """Example using JSON parser."""
    print("\n=== JSON Parser ===")
    
    parser = JSONOutputParser()
    
    prompt = f"""Extract the following information as JSON:

Person: Alice Johnson, 28 years old, Data Scientist
Skills: Python, Machine Learning, SQL

{parser.get_format_instructions()}
"""
    
    llm = ChatOpenAI(temperature=0)
    response = llm.invoke(prompt)
    
    print(f"LLM Response:\n{response.content}\n")
    
    # Parse the response
    data = parser.parse(response.content)
    
    print(f"Parsed JSON:")
    for key, value in data.items():
        print(f"  {key}: {value}")
    
    return data


def example_structured_parser():
    """Example using structured output parser."""
    print("\n=== Structured Output Parser ===")
    
    # Define response schemas
    response_schemas = [
        ResponseSchema(name="company", description="The name of the company"),
        ResponseSchema(name="industry", description="The industry the company operates in"),
        ResponseSchema(name="founded", description="The year the company was founded"),
        ResponseSchema(name="employees", description="Approximate number of employees"),
    ]
    
    parser = StructuredOutputParser(response_schemas=response_schemas)
    
    prompt = f"""Extract company information from this text:

TechCorp is a leading software company in the technology industry. 
Founded in 2010, it now employs approximately 5,000 people worldwide.

{parser.get_format_instructions()}
"""
    
    llm = ChatOpenAI(temperature=0)
    response = llm.invoke(prompt)
    
    print(f"LLM Response:\n{response.content}\n")
    
    # Parse the response
    data = parser.parse(response.content)
    
    print(f"Parsed Structured Data:")
    for key, value in data.items():
        print(f"  {key}: {value}")
    
    return data


def example_structured_parser_from_dict():
    """Example creating structured parser from dictionary."""
    print("\n=== Structured Parser from Dictionary ===")
    
    # Define schema as dictionary
    schema_dict = {
        "book_title": "The title of the book",
        "author": "The author's name",
        "genre": "The genre of the book",
        "publication_year": "The year the book was published",
        "pages": "Number of pages in the book"
    }
    
    parser = StructuredOutputParser.from_schema_dict(schema_dict)
    
    prompt = f"""Extract book information:

"The Great Gatsby" by F. Scott Fitzgerald is a classic American novel 
published in 1925. This literary fiction masterpiece has 180 pages.

{parser.get_format_instructions()}
"""
    
    llm = ChatOpenAI(temperature=0)
    response = llm.invoke(prompt)
    
    data = parser.parse(response.content)
    
    print(f"Parsed Book Information:")
    for key, value in data.items():
        print(f"  {key}: {value}")
    
    return data


def example_movie_review_parser():
    """Example parsing movie reviews with sentiment."""
    print("\n=== Movie Review Parser ===")
    
    parser = PydanticOutputParser(pydantic_object=MovieReview)
    
    prompt = f"""Analyze this movie review:

"Inception is a mind-bending masterpiece! Christopher Nolan delivers an 
incredible story about dreams within dreams. The visual effects are stunning, 
and Leonardo DiCaprio's performance is outstanding. I'd give it a 9/10 and 
highly recommend it to anyone who loves thought-provoking cinema."

{parser.get_format_instructions()}
"""
    
    llm = ChatOpenAI(temperature=0)
    response = llm.invoke(prompt)
    
    review = parser.parse(response.content)
    
    print(f"Parsed Movie Review:")
    print(f"  Movie: {review.movie_title}")
    print(f"  Rating: {review.rating}/10")
    print(f"  Sentiment: {review.sentiment}")
    print(f"  Summary: {review.summary}")
    print(f"  Would Recommend: {review.would_recommend}")
    
    return review


def example_list_extraction():
    """Example extracting lists of items."""
    print("\n=== List Extraction ===")
    
    class ShoppingList(BaseModel):
        """A shopping list with items and quantities."""
        items: List[str] = Field(description="List of items to buy")
        quantities: List[int] = Field(description="Quantity for each item")
        total_items: int = Field(description="Total number of different items")
    
    parser = PydanticOutputParser(pydantic_object=ShoppingList)
    
    prompt = f"""Extract the shopping list:

I need to buy:
- 3 apples
- 2 loaves of bread
- 1 gallon of milk
- 5 bananas
- 2 boxes of cereal

{parser.get_format_instructions()}
"""
    
    llm = ChatOpenAI(temperature=0)
    response = llm.invoke(prompt)
    
    shopping_list = parser.parse(response.content)
    
    print(f"Parsed Shopping List:")
    print(f"  Total Items: {shopping_list.total_items}")
    print(f"  Items:")
    for item, qty in zip(shopping_list.items, shopping_list.quantities):
        print(f"    - {qty}x {item}")
    
    return shopping_list


if __name__ == "__main__":
    print("=" * 60)
    print("Output Parser Examples")
    print("=" * 60)
    
    try:
        # Pydantic parser examples
        example_pydantic_parser_person()
        example_pydantic_parser_product()
        example_movie_review_parser()
        example_list_extraction()
        
        # JSON parser example
        example_json_parser()
        
        # Structured parser examples
        example_structured_parser()
        example_structured_parser_from_dict()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

# Made with Bob
