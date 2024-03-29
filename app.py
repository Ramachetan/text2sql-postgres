import time
import streamlit as st
from vertexai.generative_models import FunctionDeclaration, GenerativeModel, Part, Tool
import psycopg2
import os
from dotenv import load_dotenv
load_dotenv()


conn = psycopg2.connect(
                        dbname= os.environ.get("DB_NAME"),
                        user=os.environ.get("DB_USER"),
                        password=os.environ.get("DB_PASS"),
                        host=os.environ.get("INSTANCE_HOST"),
                        port=os.environ.get("DB_PORT")
                    )

list_datasets_func = FunctionDeclaration(
    name="list_datasets",
    description="Get a list of datasets that will help answer the user's question",
    parameters={
        "type": "object",
        "properties": {},
    },
)

list_tables_func = FunctionDeclaration(
    name="list_tables",
    description="List tables in a database that will help answer the user's question",
    parameters={
        "type": "object",
        "properties": {},
    },
)



get_table_func = FunctionDeclaration(
    name="get_table",
    description="Get information about a table, such as the columns and data types, that will help answer the user's question",
    parameters={
        "type": "object",
        "properties": {
            "table_id": {
                "type": "string",
                "description": "Fully qualified ID of the table to get information about",
            }
        },
        "required": [
            "table_id",
        ],
    },
)

sql_query_func = FunctionDeclaration(
    name="sql_query",
    description="Get information from data in Postgres using SQL queries",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "SQL query on a single line that will help give quantitative answers to the user's question when run on a Postgres SQL database and table. In the SQL query, always use the fully qualified dataset and table names.",
            }
        },
        "required": [
            "query",
        ],
    },
)

sql_query_tool = Tool(
    function_declarations=[
        list_datasets_func,
        list_tables_func,
        get_table_func,
        sql_query_func,
    ],
)

model = GenerativeModel(
    "gemini-1.0-pro",
    generation_config={"temperature": 0},
    tools=[sql_query_tool],
)

st.set_page_config(
    page_title="SQL Talk with Postgres SQL",
    page_icon="vertex-ai.png",
    layout="wide",
)

col1, col2 = st.columns([4, 1])
with col1:
    st.title("SQL Talk with Postgres SQL")
with col2:
    st.image("gcp.png")

st.subheader("Powered by Function Calling in Gemini")

with st.expander("Sample prompts", expanded=True):
    st.write(
        """
        - How many Employees are there in the database?
        - What are the tables in the database?
        - What are the columns in the Employees table?
        - What is the average salary of employees in the database?
        - In Which state do most employees live?
    """
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"].replace("$", "\$"))  # noqa: W605
        try:
            with st.expander("Function calls, parameters, and responses"):
                st.markdown(message["backend_details"])
        except KeyError:
            pass

if prompt := st.chat_input("Ask me about information in the database..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        chat = model.start_chat()
        prompt += """
            Please give a concise, high-level summary followed by detail in
            plain language about where the information in your response is
            coming from in the Postgres SQL database. Only use information that you learn
            from Postgres SQL database, do not make up information.
            """

        response = chat.send_message(prompt)
        response = response.candidates[0].content.parts[0]

        print(response)

        api_requests_and_responses = []
        backend_details = ""

        function_calling_in_process = True
        while function_calling_in_process:
            try:
                params = {}
                for key, value in response.function_call.args.items():
                    params[key] = value

                print(response.function_call.name)
                print(params)
                
                if response.function_call.name == "list_datasets":
                    cur = conn.cursor()
                    cur.execute("SELECT table_schema FROM information_schema.tables WHERE table_schema NOT IN ('pg_catalog', 'information_schema');")
                    datasets = cur.fetchall()
                    api_response = str([dataset[0] for dataset in datasets])
                    api_requests_and_responses.append(
                        [response.function_call.name, params, api_response]
                    )
                
                if response.function_call.name == "list_tables":
                    cur = conn.cursor()
                    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
                    tables = cur.fetchall()
                    api_response = str([table[0] for table in tables])
                    api_requests_and_responses.append(
                        [response.function_call.name, params, api_response]
                    )
                
                if response.function_call.name == "get_table":
                    cur = conn.cursor()
                    cur.execute(f"SELECT * FROM {params['table_id']};")
                    table_info = cur.fetchall()
                    api_response = str(table_info)
                    api_requests_and_responses.append(
                        [response.function_call.name, params, api_response]
                    )
                
                if response.function_call.name == "sql_query":
                    cur = conn.cursor()
                    cleaned_query = params["query"].replace(";", "").replace("\n", " ").replace("`", "\"")
                    cur.execute(cleaned_query)
                    try:
                        cur.execute(cleaned_query)
                        query_result = cur.fetchall()
                        api_response = str(query_result)
                    except Exception as e:
                        api_response = str(e)
                    finally:
                        api_requests_and_responses.append(
                            [response.function_call.name, params, api_response]
                        )

                print(api_response)

                response = chat.send_message(
                    Part.from_function_response(
                        name=response.function_call.name,
                        response={
                            "content": api_response,
                        },
                    ),
                )
                response = response.candidates[0].content.parts[0]

                backend_details += "- Function call:\n"
                backend_details += (
                    "   - Function name: ```"
                    + str(api_requests_and_responses[-1][0])
                    + "```"
                )
                backend_details += "\n\n"
                backend_details += (
                    "   - Function parameters: ```"
                    + str(api_requests_and_responses[-1][1])
                    + "```"
                )
                backend_details += "\n\n"
                backend_details += (
                    "   - API response: ```"
                    + str(api_requests_and_responses[-1][2])
                    + "```"
                )
                backend_details += "\n\n"
                with message_placeholder.container():
                    st.markdown(backend_details)

            except AttributeError:
                function_calling_in_process = False

        time.sleep(3)

        full_response = response.text
        with message_placeholder.container():
            st.markdown(full_response.replace("$", "\$"))  # noqa: W605
            with st.expander("Function calls, parameters, and responses:"):
                st.markdown(backend_details)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
                "backend_details": backend_details,
            }
        )