from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_PROMPT

# Table Info Generation Prompt -
TABLE_INFO_PROMPT = """
Give me a summary of the table with the following JSON format:

{
  "table_name": "...",
  "table_summary": "...",
  "column_descriptions": {
    "column1": "description...",
    "column2": "description...",
    ...
  }
}

Instructions:
- The table name must be unique to the table and describe its primary content while being concise. Use Mongolian names if appropriate for clarity.
- Do NOT output a generic table name (e.g. table, my_table, data).
- For each column, provide a detailed description.
  - Describe the **type of data** it contains (e.g., numeric, text, date, code, descriptive label).
  - Explain its **purpose or meaning** within the table.
  - If the column name is cryptic (e.g., `dtval_co`, `scr_mn`, `code1`), **infer its full semantic meaning** based on other columns and the provided sample data.
  - **For `dtval_co`**: Describe it as the primary numeric value, and critically state that its unit or currency (e.g., Mongolian Togrog, USD) is defined by the `scr_mn1` or `scr_eng1` columns.
  - **For `code1`, `scr_mn1`, `scr_eng1`**: These columns are essential for defining the *unit of measurement or currency* for the `dtval_co` value. Describe them specifically in this context. For `code1`, mention specific values like '1' for Togrog and '2' for USD, if evident from data. For `scr_mn1`/`scr_eng1`, mention they provide the descriptive name of the unit/currency (e.g., '1 хүнд ногдох ДНБ, мян.төг' for thousands of Togrog, 'GDP per capita, USD' for US Dollars).
  - **Crucially, explain relationships**: If columns work together to define a unique record or describe a value (e.g., `period` + `code` + `code1` defines a specific metric's value), explicitly state this relationship.
  - For descriptive columns (`scr_mn`, `scr_eng`), state what they are describing (e.g., "This describes the type of price (current, constant) for the GDP value.").
- Pay special attention to columns that are essential for filtering or uniquely identifying rows (e.g., `period`, `code`, `code1`).
- Output column_descriptions for **all** columns that appear in the 'Table Structure' or 'Sample Data'. **If a column like `code2` or `scr_mn2` is NOT present in the 'Sample Data', you can omit it or state it's not applicable based on current data to avoid confusion.**
- Do NOT make the table name one of the following: {exclude_table_name_list}

Table Name: {table_name}
Table Structure: {table_structure}
Sample Data:
{table_data}

Summary: """

# Response Synthesis Prompt 
RESPONSE_SYNTHESIS_PROMPT = """
Given an input question, synthesize a response from the query results in Mongolian language.
Instructions:
- Answer in clear, natural Mongolian.
- Be concise but informative.
- **When interpreting numerical data (from 'dtval_co'), ALWAYS use the precise unit/currency described in the 'scr_mn1' column (e.g., 'мян.төг' for thousands of Togrog, 'ам доллар' for USD). Do NOT invent units like 'тэрбум' or 'мянга' if they are not explicitly specified as the unit in 'scr_mn1' or 'scr_eng1'.**
- If 'scr_mn1' is not available, use 'scr_eng1' or infer the unit from the context provided by other columns.
- When referring to the metric, use the exact Mongolian description from 'scr_mn1' (e.g., '1 хүнд ногдох ДНБ' instead of just 'ДНБ' or 'нийт бүтээгдэхүүн').
- If multiple rows show similar data with different codes or identifiers, present all relevant information clearly, grouping related data logically by period, price type, and unit.
- If no data is found, explain that politely in Mongolian.
- Always include context from descriptive columns when presenting numerical values.
Query: {query_str}
SQL: {sql_query}
SQL Response: {context_str}
Response: """

# Enhanced Text-to-SQL Prompt with better multi-column guidance 
MONGOLIAN_TEXT_TO_SQL_PROMPT = """
Given an input question in Mongolian or English, create a syntactically correct {dialect} query to run.

CRITICAL INSTRUCTIONS:
- Understand the question even if it's in Mongolian language
- Use double quotes around table names and column names if they contain special characters
- IMPORTANT: Many data records require MULTIPLE columns in the WHERE clause to uniquely identify the correct row
- Pay special attention to combination columns like (period, code, scr_mn/scr_eng) that work together
- When filtering data, consider ALL relevant columns that define the specific record you're looking for
- Look at the column descriptions carefully to understand what each column represents
- For questions about specific years, prices, currencies, etc., make sure to filter by ALL relevant identifying columns
- **Ensure the SELECT clause includes both `dtval_co` and the unit descriptor columns (`scr_mn1`, `scr_eng1`) when the question pertains to a numerical value with a unit or currency.**

Available tables and columns with descriptions:
{schema}

Question: {query_str}
SQLQuery: """

class PromptManager:
    """Manages all prompt templates for the chatbot"""
    
    def __init__(self, dialect: str = "postgresql"):
        self.dialect = dialect
        self._initialize_prompts()
    
    def _initialize_prompts(self):
        """Initialize all prompt templates"""
        # Text-to-SQL prompt
        self.text2sql_prompt = PromptTemplate(MONGOLIAN_TEXT_TO_SQL_PROMPT).partial_format(
            dialect=self.dialect
        )
        
        # Response synthesis prompt
        self.response_synthesis_prompt = PromptTemplate(RESPONSE_SYNTHESIS_PROMPT)
        
        # Table info prompt
        self.table_info_prompt = PromptTemplate(TABLE_INFO_PROMPT)
    
    def get_text2sql_prompt(self) -> PromptTemplate:
        """Get the text-to-SQL prompt"""
        return self.text2sql_prompt
    
    def get_response_synthesis_prompt(self) -> PromptTemplate:
        """Get the response synthesis prompt"""
        return self.response_synthesis_prompt
    
    def get_table_info_prompt(self) -> PromptTemplate:
        """Get the table info generation prompt"""
        return self.table_info_prompt
    
    def format_table_info_prompt(self, table_name: str, table_structure: str, 
                                table_data: str, exclude_list: list = None) -> str:
        """Format the table info prompt"""
        if exclude_list is None:
            exclude_list = []
             
        return self.table_info_prompt.format(
            table_name=table_name,
            table_structure=table_structure,
            table_data=table_data,
            exclude_table_name_list=str(exclude_list)
        )

prompt_manager = PromptManager()